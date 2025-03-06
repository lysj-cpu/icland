"""Trajectory manager server code."""

import threading
import time
from multiprocessing import Event, Process
from typing import Any

import numpy as np
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit


class _TrajectoryManagerBackend:  # pragma: no cover
    """The trajectory manager server backend."""

    def __init__(self, max_steps: int = 1000):
        """Initialise the trajectory manager backend."""
        self.simulation_data: list[Any] = []  # List to store all timesteps
        self.max_steps = max_steps
        self.simulation_ended = False  # Flag to track simulation status

    def append_simulation_data(
        self,
        timestep: int,
        positions: list[Any] | np.ndarray[tuple[int, ...], np.dtype[Any]],
        rewards: list[Any],
        image: str,
    ) -> None:
        """Append a new simulation timestep with all data."""
        if len(self.simulation_data) >= self.max_steps:
            self.simulation_data.pop(0)  # Remove oldest data if limit reached
        self.simulation_data.append(
            {
                "timestep": timestep,
                "positions": positions,
                "rewards": rewards,
                "image": image,  # Base64 encoded string
            }
        )

    def get_simulation_data(self, timestep: int | None = None) -> list[Any] | None:
        """Retrieve simulation data for a specific timestep or all data."""
        if timestep is not None:
            return next(
                (data for data in self.simulation_data if data["timestep"] == timestep),
                None,
            )
        return self.simulation_data

    def set_simulation_ended(self) -> None:
        """Mark the simulation as ended."""
        self.simulation_ended = True


class _TrajectoryManagerServer:  # pragma: no cover
    """The trajectory manager server controller."""

    def __init__(self, max_steps: int = 1000, shutdown_event: Any = None):
        """Initialise the trajectory manager server controller."""
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.trajectory_mgr = _TrajectoryManagerBackend(max_steps)
        self.shutdown_event = shutdown_event or Event()
        self.setup_routes()
        self.setup_socket_events()

    def setup_routes(self) -> None:
        """Setup the server route."""

        @self.app.route("/")
        def index() -> str:
            return render_template("index.html")

    def setup_socket_events(self) -> None:
        """Setup the server socket.IO event handlers."""

        @self.socketio.on("update_simulation_data")  # type: ignore
        def handle_simulation_update(data: dict[str, Any]) -> None:
            timestep = data["timestep"]
            positions = data["positions"]
            rewards = data["rewards"]
            image = data["image"]
            self.trajectory_mgr.append_simulation_data(
                timestep, positions, rewards, image
            )
            emit(
                "simulation_data_update",
                {
                    "timestep": timestep,
                    "positions": positions,
                    "rewards": rewards,
                    "image": image,
                },
                broadcast=True,
            )

        @self.socketio.on("end_simulation")  # type: ignore
        def handle_end_simulation() -> None:
            self.trajectory_mgr.set_simulation_ended()
            emit("simulation_ended", broadcast=True)
            print("Received end simulation request from client")

        @self.socketio.on("get_simulation_data")  # type: ignore
        def handle_get_simulation_data(data: dict[str, Any] | None = None) -> None:
            timestep = data.get("timestep") if data else None
            simulation_data = self.trajectory_mgr.get_simulation_data(timestep)
            emit("simulation_data_response", simulation_data)

        @self.socketio.on("stop_server")  # type: ignore
        def handle_stop_server() -> None:
            print("Received stop server request from client")
            emit("server_stopped", broadcast=True)  # Notify all clients
            self.shutdown_event.set()  # Signal shutdown

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
        """Main code to run the server on a separate thread."""
        server_thread = threading.Thread(
            target=self.socketio.run,
            args=(self.app,),
            kwargs={
                "host": host,
                "port": port,
                "debug": debug,
                "allow_unsafe_werkzeug": True,
            },
            daemon=True,
        )
        server_thread.start()

        # Monitor the shutdown event
        while not self.shutdown_event.is_set():
            time.sleep(0.1)

        # Clean shutdown
        self.socketio.stop()
        server_thread.join(timeout=2)
        if server_thread.is_alive():
            print("Server thread did not terminate gracefully")
        else:
            print("Server shut down cleanly")


class TrajectoryManager:  # pragma: no cover
    """Trajectory monitor entry point."""

    def __init__(self, max_steps: int = 1000):
        """Initialise the trajectory monitor entry point."""
        self.shutdown_event = Event()
        self.max_steps = max_steps
        self.server = _TrajectoryManagerServer(
            max_steps=self.max_steps, shutdown_event=self.shutdown_event
        )
        self.server_process = Process(target=self.server.run, daemon=False)

    def start(self) -> None:
        """Start the trajectory manager server.

        This method initializes the server by clearing the shutdown event and
        starting the server process in a separate process. The server will begin
        accepting connections and handling simulation data.
        """
        self.shutdown_event.clear()
        self.server_process.start()

    def stop(self) -> None:
        """Stop the trajectory manager server.

        This method signals the server to shut down by setting the shutdown event
        and waits for the server process to terminate gracefully within a 5-second
        timeout. If the process does not terminate within this period, it is forcibly
        terminated, and a message is printed to indicate the outcome.
        """
        self.shutdown_event.set()
        self.server_process.join(timeout=5)
        if self.server_process.is_alive():
            print("Server process did not terminate gracefully, forcing termination.")
            self.server_process.terminate()
        else:
            print("Server process terminated cleanly")
