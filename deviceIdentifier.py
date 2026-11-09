
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import json
from threading import Lock

app = FastAPI(title="Sensor Backend")

DATA_FILE = Path(__file__).parent / "received_data.json"
_data_lock = Lock()


class SensorRecord(BaseModel):
    deviceId: str = Field(min_length=1)
    device: str = ""
    name: str = ""
    sensorId: str
    data: list[float]
    timestamp: int | None = None
    friendlyTimeStamp: str | None = None


@app.post("/upload")
def upload(records: list[SensorRecord]):
    if not records:
        raise HTTPException(
            status_code=400,
            detail="No records provided"
        )

    with _data_lock:
        existing = []

        if DATA_FILE.exists():
            existing = json.loads(
                DATA_FILE.read_text(encoding="utf-8")
            )

        existing.extend(
            record.model_dump() for record in records
        )

        DATA_FILE.write_text(
            json.dumps(existing, indent=2),
            encoding="utf-8"
        )

    return {
        "status": "OK",
        "saved": len(records)
    }


@app.get("/api/data")
def get_data():
    with _data_lock:
        if not DATA_FILE.exists():
            return []

        return json.loads(
            DATA_FILE.read_text(encoding="utf-8")
        )
