from asyncio.log import logger
import pydicom
from pydicom.filereader import dcmread
from pydicom.uid import UID, ExplicitVRBigEndian
from pathlib import Path
from pydantic import BaseModel

from numpy import ndarray as np_ndarray

from scipy.interpolate import RegularGridInterpolator  # type: ignore
from glob import glob
from typing import List, Any, Callable

from string import Template
import datetime

# from matplotlib import pyplot as plt

RT_PLAN_UID = "1.2.840.10008.5.1.4.1.1.481.5"
TEMPLATE_MEASURE = "./templates/omniPro_measure"
TEMPLATE_HEADER = "./templates/omniPro_header"


class Point(BaseModel):
    x: float = 0
    y: float = 0
    z: float = 0
    d: float = 0


class Line(BaseModel):
    start: Point = Point(x=-10, y=0, z=0)
    stop: Point = Point(x=10, y=0, z=0)
    delta: float = 0.1


class Config(BaseModel):
    input: str
    output: str
    ops: List[Line]


class DicomData:
    def __init__(
        self,
        filepath: str,
        ts_uid: UID = ExplicitVRBigEndian,
    ) -> None:

        dose: pydicom.FileDataset = dcmread(filepath, force=True)
        dose.file_meta.TransferSyntaxUID = ts_uid

        # The Dicom Dose file should have a referenced SOP Instance of plan data
        refUID = dose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
        # print(refUID)

        # Idealy this dicom dose should have only one beam sequence. We are only
        # trying to get the first reference
        beam = (
            dose.ReferencedRTPlanSequence[0]
            .ReferencedFractionGroupSequence[0]
            .ReferencedBeamSequence[0]
            .ReferencedBeamNumber
        )

        # File parent folder
        file_folder = str(Path(filepath).parent.absolute())

        # All present dicom files inside parent folder
        all_dcm_files = glob(file_folder + "/*.dcm")

        # All Dicom files present in parent folder
        file_set: List[pydicom.FileDataset] = [
            dcmread(fp, force=True) for fp in all_dcm_files
        ]
        # All Dicom Plan files present in parent folder
        plan_set = [fs for fs in file_set if fs.SOPClassUID == RT_PLAN_UID]
        # print([plan.PatientID for plan in plan_set])

        # The Only Dicom Plan File related to this Dicom Dose file
        plans = [pl for pl in plan_set if pl.SOPInstanceUID == refUID]

        plan = plans[0]
        # print(plan.PatientID)

        # Control Point Sequence of plan
        sequence = plan.BeamSequence[beam - 1].ControlPointSequence[0]

        # Adding data to class

        self.data: np_ndarray[float, Any] = dose.pixel_array * dose.DoseGridScaling  # type: ignore

        # Cooridnates of isocenter in patient based system in x, y, z
        iso = sequence.IsocenterPosition
        self.iso = Point(x=iso[0], y=iso[1], z=iso[2])

        # Cooridnates of surface entry in patient based system in x, y, z
        entry = sequence.SurfaceEntryPoint
        self.entry = Point(x=entry[0], y=entry[1], z=entry[2])

        # Image position related to patient in dicom format. Format is [x (col), y (row), z (frame)]
        # https://stackoverflow.com/questions/40115444/dicom-understanding-the-relationship-between-patient-position-0018-5100-image
        crf_ref = dose.ImagePositionPatient

        # Pixel spacing in format [row (y), col(x)]
        pixel = dose.PixelSpacing

        # Regular grid for interpolation in format frame (z), row (y) and col (x)
        frm_list = [(crf_ref[2] + f) for f in dose.GridFrameOffsetVector]
        row_list = [(crf_ref[1] + r) * pixel[0] for r in range(dose.Rows)]
        col_list = [(crf_ref[0] + c) * pixel[1] for c in range(dose.Columns)]

        self.interpolator: Callable[
            [list[List[float]]], np_ndarray[float, Any]
        ] = RegularGridInterpolator(
            (frm_list, row_list, col_list),
            self.data,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        date = datetime.datetime.strptime(plan.RTPlanDate, "%Y%m%d")
        self.date = date.strftime("%m-%d-%Y")

        time = datetime.datetime.strptime(plan.RTPlanTime.split(".")[0], "%H%M%S")
        self.time = time.strftime("%H:%M:%S")

        self.energy = sequence.NominalBeamEnergy

        self.ssd = sequence.SourceToSurfaceDistance
        jaw = sequence.BeamLimitingDevicePositionSequence[0].LeafJawPositions
        self.jaw = [float(j) for j in jaw]

    def get_dose(self, line: Line) -> List[Point]:

        # Creation of grid system for start and stop values
        dx = line.stop.x - line.start.x
        dy = line.stop.y - line.start.y
        dz = line.stop.z - line.start.z
        norm = (dx**2 + dy**2 + dz**2) ** 0.5
        steps = int(norm / line.delta)

        # This vector is represented in physics system.
        # X is crossline
        # Y is inline
        # Z is height measured positbly going down.
        # I know, this is not a right hand axis system... but fuck it!
        s_vec: List[float] = [st * line.delta / norm for st in range(steps)]
        x_vec = [line.start.x + st * dx for st in s_vec]
        y_vec = [line.start.y + st * dy for st in s_vec]
        z_vec = [line.start.z + st * dz for st in s_vec]

        # This vector is in patient reference and should be passed as matrix..
        # We need to massage the axis to work. Dont think... its right.
        dcm_vec = [
            [self.iso.z + y, self.iso.y + z, self.iso.x + x]
            for (x, y, z) in zip(x_vec, y_vec, z_vec)
        ]

        d_vec = [float(d) for d in self.interpolator(dcm_vec).flatten()]

        return [
            Point(x=x, y=y, z=z, d=d)
            for (x, y, z, d) in zip(x_vec, y_vec, z_vec, d_vec)
        ]


def get_omniPro_string(dcm: DicomData, line: Line, ind: int) -> str:

    fm: Callable[[float], str] = lambda x: f"{x:>7.1f}"

    dose = dcm.get_dose(line)
    max_dose = max(map(lambda x: x.d, dose))
    meas = [
        f"= \t{fm(p.x)}\t{fm(p.y)}\t{fm(p.z)}\t{fm(100*p.d/max_dose)}" for p in dose
    ]
    measures = "\n".join(meas)

    tmp = Template(open(TEMPLATE_MEASURE, "r").read())
    output = tmp.substitute(
        number=ind,
        date=dcm.date,
        time=dcm.time,
        size=int(dcm.jaw[1] - dcm.jaw[0]),
        energy=dcm.energy,
        ssd=dcm.ssd,
        points=len(dose),
        start=f"{fm(dose[0].x)}\t{fm(dose[0].y)}\t{fm(dose[0].z)}",
        stop=f"{fm(dose[-1].x)}\t{fm(dose[-1].y)}\t{fm(dose[-1].z)}",
        measures=measures,
    )

    return output


def get_omniPro_file(config: Config) -> None:

    dcm = DicomData(config.input)

    tmp = Template(open(TEMPLATE_HEADER, "r").read())
    header = tmp.substitute(measurements=len(config.ops))
    footer = f"\n:EOF # End of File\n"

    content: List[str] = []
    for ind, op in enumerate(config.ops):
        content.append(get_omniPro_string(dcm, op, ind + 1))

    with open(config.output, "w+") as f:
        f.write(header + "\n".join(content) + footer)


def create_omniPro_from_yaml(file: str) -> None:

    from yaml import Loader as yaml_Loader
    from yaml import load as yaml_load

    try:
        config = yaml_load(open(file, "r"), Loader=yaml_Loader)
        print(config)
    except ValueError as error:
        logger.error(error)
        return None

    get_omniPro_file(Config(**config))


def create_omniPro_from_jason(file: str) -> None:

    try:
        config = Config.parse_file("./data/config.json")
    except ValueError as error:
        logger.error(error)
        return None

    get_omniPro_file(config)


# if __name__ == "__main__":

#     create_omniPro_from_jason("./data/config.json")
#     create_omniPro_from_yaml("./data/config.yaml")
