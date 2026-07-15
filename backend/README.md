# NumiIT Backend

This is the FastAPI backend for the NumiIT project. It handles image processing, ML inference for coin and character recognition, and provides a REST API.

## Folder Structure

- `app/`: Contains the main application logic, API routes, models, and ML inference services.
- `alembic/`: Database migration scripts.
- `uploads/`: Temporary storage for uploaded images.
- `requirements.txt`: Python dependencies required to run the backend.

## Setup & Running

This backend can be run using the global `run.py` script from the project root:

```bash
# To install dependencies (only needed once)
python run.py setup

# To run the backend server (accessible across your local network)
python run.py backend
```

Alternatively, you can run the server directly using Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
