
from contextlib import asynccontextmanager
from threading import Lock
from pathlib import Path
import base64

import cv2
import numpy as np

from fastapi import FastAPI, Request, HTTPException


from src.dms_processor import DMSProcessor


processor = None
processor_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):

    global processor

    processor = DMSProcessor()
    app.state.processor = processor

    yield

    processor.close()
    processor = None


app = FastAPI(
    title="Driver Monitoring System API",
    lifespan=lifespan
)


@app.get("/health")
def health():

    return {
        "status": "ok",
        "message": "DMS API is running"
    }


@app.post("/process-frame")
async def process_frame(request: Request):

    image_bytes = await request.body()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Aucune image recue"
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Image invalide"
        )

    try:

        with processor_lock:

            processed_frame, metrics = (
                request.app.state.processor.process_frame(frame)
            )

        success, encoded_image = cv2.imencode(
            ".jpg",
            processed_frame
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Erreur encodage JPEG"
            )

        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")

        return {
            "image": image_base64,
            "metrics": metrics
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur traitement DMS : {str(error)}"
        )


@app.post("/calibrate")
def calibrate(request: Request):

    with processor_lock:

        success = request.app.state.processor.calibrate()

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Aucune position du visage disponible pour calibrer"
        )

    return {
        "status": "success",
        "message": "Head Pose calibrated"
    }


@app.post("/reset-calibration")
def reset_calibration(request: Request):

    with processor_lock:

        request.app.state.processor.reset_calibration()

    return {
        "status": "success",
        "message": "Head Pose calibration reset"
    }
