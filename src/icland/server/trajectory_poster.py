"""Trajectory poster code."""

import base64
import threading
import time
import uuid
from typing import Any, cast

import cv2
import jax
import numpy as np
import socketio

from icland.renderer.renderer import render_top_down
from icland.types import *


class WebSocketTrajectoryPoster:  # pragma: no cover
    """Trajectory poster entry point."""

    def __init__(
        self, server_url: str = "http://localhost:5000", agent_id: str | None = None
    ):
        """Initializes a poster entry point to the server."""
        self.sio = socketio.Client()
        self.server_url = server_url
        self.agent_id = agent_id or str(uuid.uuid4())
        self.stop_event = threading.Event()
        self.setup_socket_events()

    def setup_socket_events(self) -> None:
        """Sets up socket.IO events."""

        @self.sio.event  # type: ignore
        def connect() -> None:
            print(f"Connected to server as agent {self.agent_id}")

        @self.sio.on("simulation_data_update")  # type: ignore
        def on_simulation_data_update(data: Any) -> None:
            print(
                f"Server confirmed simulation data update for timestep {data['timestep']}"
            )

        @self.sio.on("simulation_ended")  # type: ignore
        def on_simulation_ended() -> None:
            print("Server confirmed simulation has ended")

    def connect(self) -> None:
        """Establish a connection to the server.

        Attempts to connect to the server specified by `server_url` with a maximum
        of 5 retries. If a connection attempt fails, it waits 1 second before the
        next attempt. Raises a `ConnectionError` if all retries are exhausted.

        Raises:
            ConnectionError: If connection fails after all retry attempts.
        """
        max_retries = 5
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                self.sio.connect(self.server_url)
                return
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        raise ConnectionError("Failed to connect to server after retries")

    def post_simulation_data(
        self,
        timestep: int,
        positions: np.ndarray[tuple[int, ...], np.dtype[Any]],
        rewards: list[Any],
        image: str,
    ) -> None:
        """Post raw simulation data for a specific timestep to the server.

        Constructs a payload containing the timestep, agent positions, rewards,
        and an image, then emits it to the server via the 'update_simulation_data'
        event. Prints a confirmation message upon successful posting.

        Args:
            timestep (int): The simulation timestep.
            positions (np.ndarray): Array of agent positions, can be multi-dimensional.
            rewards (list[Any]): List of rewards for each agent.
            image (str): Base64-encoded image string representing the simulation frame.
        """
        payload = {
            "timestep": timestep,
            "positions": [
                pos.tolist() if isinstance(pos, np.ndarray) else pos
                for pos in positions
            ],
            "rewards": rewards,
            "image": image,  # Assuming base64 encoded
        }
        self.sio.emit("update_simulation_data", payload)
        print(f"Posted simulation data for timestep {timestep}")

    def end_simulation(self) -> None:
        """Signal the server that the simulation has ended.

        Emits an 'end_simulation' event to the server to indicate the completion
        of the simulation and prints a confirmation message.
        """
        self.sio.emit("end_simulation")
        print("Posted end of simulation signal")

    def stop(self) -> None:
        """Signal the server to terminate its operation.

        Sets the `stop_event` to indicate that the server should shut down. This
        does not directly disconnect the client but prepares for termination.
        """
        self.stop_event.set()

    def __numpy_to_base64(
        self,
        image_array: np.ndarray[tuple[int, ...], np.dtype[Any]],
        window_size: tuple[int, int],
    ) -> str:
        """Convert a NumPy array to a Base64-encoded PNG string."""
        image_array = (image_array * 0xFF).astype(np.uint8)

        image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        image_cv2 = cv2.resize(image_cv2, window_size, interpolation=cv2.INTER_NEAREST)
        _, buffer = cv2.imencode(".png", image_cv2)
        base64_string = base64.b64encode(buffer).decode("utf-8")
        return base64_string

    def post_icland_data(
        self,
        timestep: int,
        icland_params: ICLandParams,
        icland_state: ICLandState,
        icland_actions: jax.Array,
        icland_rewards: jax.Array,
        window_size: tuple[int, int] = (960, 720),
    ) -> None:
        """Post ICLand-specific simulation data for a timestep to the server.

        Processes ICLand simulation data, including parameters and state, to extract
        agent positions and render a top-down view. Converts the rendered frame to
        a Base64-encoded string and posts it along with positions and rewards to
        the server using `post_simulation_data`.

        Args:
            timestep (int): The simulation timestep.
            icland_params (ICLandParams): Parameters defining the ICLand simulation.
            icland_state (ICLandState): Current state of the ICLand simulation.
            icland_actions (jax.Array): Actions taken by agents in this timestep.
            icland_rewards (jax.Array): Rewards received by agents in this timestep.
            window_size (tuple[int, int], optional): Size of the rendered frame in pixels.
                Defaults to (960, 720).

        Returns:
            None
        """
        # Unpack state
        mjx_data = icland_state.mjx_data
        agent_variables = icland_state.agent_variables
        prop_variables = icland_state.prop_variables

        # Unpack params
        agent_info = icland_params.agent_info
        prop_info = icland_params.prop_info
        world = icland_params.world

        # Get actual agent count, and hence positions
        agent_count = agent_info.agent_count
        agent_positions = jax.vmap(lambda bid: mjx_data.xpos[bid])(agent_info.body_ids)[
            :agent_count
        ]

        frame = render_top_down(
            agent_info,
            agent_variables,
            prop_info,
            prop_variables,
            icland_actions,
            world,
            mjx_data,
        )

        frame_np = np.nan_to_num(np.array(frame))
        frame_enc = self.__numpy_to_base64(frame_np, window_size)
        data = {
            "timestep": timestep,
            "positions": np.array(agent_positions).round(2),
            "rewards": icland_rewards[:agent_count].tolist(),
            "image": frame_enc,
        }
        self.post_simulation_data(
            cast(int, data["timestep"]),
            cast(np.ndarray[tuple[int, ...], np.dtype[Any]], data["positions"]),
            cast(list[Any], data["rewards"]),
            cast(str, data["image"]),
        )
