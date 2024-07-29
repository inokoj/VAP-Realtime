import argparse
import csv
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

static_path = Path(__file__).parent.absolute() / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

left_audio_file_path = "left_speaker_audio.wav"
right_audio_file_path = "right_speaker_audio.wav"
prediction_file_path = "prediction.csv"


@app.get("/")
def read_root():
    return FileResponse(static_path / "index.html")


@app.get("/audio/left")
def get_left_audio():
    return FileResponse(left_audio_file_path, media_type="audio/wav")


@app.get("/audio/right")
def get_right_audio():
    return FileResponse(right_audio_file_path, media_type="audio/wav")


@app.get("/data")
def get_data():
    data = []
    with open(prediction_file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filtered_row = {
                "time_sec": row["time_sec"],
                "p_now(0=left)": row["p_now(0=left)"],
                "p_future(0=left)": row["p_future(0=left)"],
                "p_now(1=right)": row["p_now(1=right)"],
                "p_future(1=right)": row["p_future(1=right)"],
            }
            data.append(filtered_row)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI server with specified audio and prediction files.")
    parser.add_argument(
        "--left_audio", type=str, default=left_audio_file_path, help="Path to the left speaker audio file"
    )
    parser.add_argument(
        "--right_audio", type=str, default=right_audio_file_path, help="Path to the right speaker audio file"
    )
    parser.add_argument("--prediction", type=str, default=prediction_file_path, help="Path to the prediction file")

    args = parser.parse_args()
    left_audio_file_path = args.left_audio
    right_audio_file_path = args.right_audio
    prediction_file_path = args.prediction

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
