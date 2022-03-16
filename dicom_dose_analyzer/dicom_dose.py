from asyncio.log import logger
import pydicom
from pydicom.filereader import dcmread
from pydicom.uid import UID, ExplicitVRBigEndian
from pathlib import Path
from pydantic import BaseModel

from yaml import Loader as yaml_Loader
from yaml import load as yaml_load
from json import load as json_load

from numpy import ndarray as np_ndarray

from scipy.interpolate import RegularGridInterpolator  # type: ignore
from glob import glob
from typing import List, Any, Callable, Union

from string import Template
import datetime

RT_PLAN_UID = "1.2.840.10008.5.1.4.1.1.481.5"
TEMPLATE_MEASURE = "./dicom_dose_analyzer/templates/omniPro_measure"
TEMPLATE_HEADER = "./dicom_dose_analyzer/templates/omniPro_header"


def get_beam_number(dose: pydicom.FileDataset, plan: pydicom.FileDataset) -> int:
    if "ReferencedFractionGroupSequence" in dose.ReferencedRTPlanSequence[0]:
        beam_number = (
            dose.ReferencedRTPlanSequence[0]
            .ReferencedFractionGroupSequence[0]
            .ReferencedBeamSequence[0]
            .ReferencedBeamNumber
        )
    else:
        beam_number = 1

    for ind, seq in enumerate(plan.FractionGroupSequence[0].ReferencedBeamSequence):
        if seq.ReferencedBeamNumber == beam_number:
            beam_number = ind + 1

    return beam_number


def get_field_size(beam_sequence: Any) -> float:

    if "ApplicatorSequence" in beam_sequence:
        ap_id = beam_sequence.ApplicatorSequence[0].ApplicatorID
        if "X" in ap_id:
            size = 10 * float(ap_id.split("X")[0])
        else:
            size = 10 * float(ap_id.split(" ")[0])

    else:
        control_sequence = beam_sequence.ControlPointSequence[0]
        jaw = control_sequence.BeamLimitingDevicePositionSequence[0].LeafJawPositions
        size = float(jaw[1]) - float(jaw[0])

    return size


class Point(BaseModel):
    x: float = 0
    y: float = 0
    z: float = 0
    d: float = 0


class Line(BaseModel):
    name: str = ""
    ref: str = "surface"
    start: Point = Point(x=-10, y=0, z=0)
    stop: Point = Point(x=10, y=0, z=0)
    delta: float = 0.1


class Config(BaseModel):
    files: List[str]
    ops: List[Line]
    folder: Union[str, None]


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

        if not plans:
            print("Missing dicom plan file")
            return None

        plan = plans[0]

        # Idealy this dicom dose should have only one beam sequence. We are only
        # trying to get the first reference
        beam = get_beam_number(dose, plan)

        # Control Point Sequence of plan
        sequence = plan.BeamSequence[beam - 1].ControlPointSequence[0]

        # Adding data to class

        self.data: np_ndarray[float, Any] = dose.pixel_array * dose.DoseGridScaling  # type: ignore

        # Cooridnates of isocenter in patient based system in x, y, z
        iso = sequence.IsocenterPosition
        self.iso = Point(x=iso[0], y=iso[1], z=iso[2])

        # Cooridnates of surface entry in patient based system in x, y, z
        surface = sequence.SurfaceEntryPoint
        self.surface = Point(x=surface[0], y=surface[1], z=surface[2])

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
        self.field_size = get_field_size(plan.BeamSequence[beam - 1])

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

        # Depending on reference of line dose where to setup initial position
        if line.ref == "surface":
            ref = self.surface
        elif line.ref == "isocenter":
            ref = self.iso
        else:
            ref = self.surface

        # This vector is in patient reference and should be passed as matrix..
        # That means -> (dcm_z + vec_y, dcm_y + vec_z, dcm_x + vec_x)

        # We need to massage the axis to work. Dont think... its right.
        dcm_vec = [
            [ref.z + y, ref.y + z, ref.x + x] for (x, y, z) in zip(x_vec, y_vec, z_vec)
        ]

        d_vec = [float(d) for d in self.interpolator(dcm_vec).flatten()]

        # The other problem its on the reference system of IBA. By default uses
        # 0 degree system so we need to invert x and y. x -> -y, y -> x
        return [
            Point(x=-y, y=x, z=z, d=d)
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
        size=int(dcm.field_size),
        energy=dcm.energy,
        ssd=dcm.ssd,
        points=len(dose),
        start=f"{fm(dose[0].x)}\t{fm(dose[0].y)}\t{fm(dose[0].z)}",
        stop=f"{fm(dose[-1].x)}\t{fm(dose[-1].y)}\t{fm(dose[-1].z)}",
        measures=measures,
    )

    return output


def create_config(file: str) -> Union[Config, None]:

    file_suffix = Path(file).suffix

    try:
        if file_suffix == ".yaml":
            config = yaml_load(open(file, "r"), Loader=yaml_Loader)
        elif file_suffix == ".json":
            config = json_load(open(file, "r"))
        else:
            logger.error(f"Cannot find type {file_suffix} in parser")
            return None

    except ValueError as error:
        logger.error(error)
        return None

    return Config(**config)


def create_omniPro_file(file: str) -> None:

    config = create_config(file)

    if not config:
        logger.error(f"Could not initialize configuration for file: {file}")
        return None

    for dcm_file in config.files:
        dcm = DicomData(dcm_file)

        tmp = Template(open(TEMPLATE_HEADER, "r").read())
        header = tmp.substitute(measurements=len(config.ops))
        footer = f"\n:EOF # End of File\n"

        content: List[str] = []
        for ind, op in enumerate(config.ops):
            content.append(get_omniPro_string(dcm, op, ind + 1))

        if config.folder:
            output = Path(config.folder) / Path(dcm_file).with_suffix(".asc").name
        else:
            output = Path(dcm_file).with_suffix(".asc")

        with open(output, "w+") as f:
            f.write(header + "\n".join(content) + footer)


# if __name__ == "__main__":

#     # create_omniPro_from_json("./data/config.json")
#     create_omniPro_file("./data/config.yaml")
