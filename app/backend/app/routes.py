from pathlib import Path
import shutil
from typing import Dict, Any
from flask import jsonify, Flask, request

from utils import EyeTracker, DataAnalyzer
from utils import SessionManager

eye_tracker = None
mgr = SessionManager(parent_dir=Path("data"))


def init_routes(app: Flask) -> None:
    @app.route("/init", methods=["POST"])
    def init() -> Dict[str, Any]:
        global eye_tracker, mgr
        try:
            data = request.get_json()
            fs = data.get("focus_session")
            mgr.init_session(focus_session=fs)
            eye_tracker = EyeTracker(mgr.session.path["raw"], SessionManager.RESOLUTION)
            return (
                jsonify(
                    {
                        "message": f"Session initialized. Focus session={mgr.focus_session}."
                    }
                ),
                200,
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/clear", methods=["POST"])
    def clear() -> Dict[str, Any]:
        global mgr
        folder_path = mgr.parent_dir

        if folder_path.exists():
            try:
                for item in folder_path.iterdir():
                    if item.is_file():
                        item.unlink()  # delete file
                    elif item.is_dir():
                        shutil.rmtree(item)  # delete directory recursively
                return jsonify({"message": "Data folder cleared."}), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Directory not found."}), 404

    @app.route("/switch", methods=["POST"])
    def switch_view() -> Dict[str, Any]:
        global mgr
        data = request.get_json()
        # extract currentPage from request body
        current_page = data.get("currentPage")
        mgr.switch_view(current_page=current_page)
        return (
            jsonify(
                {
                    "message": f"No.{mgr.switch_cnt} switch logged. View name = {current_page}"
                }
            ),
            200,
        )

    @app.route("/start", methods=["POST"])
    def start() -> Dict[str, Any]:
        global eye_tracker
        if eye_tracker is None:
            return jsonify({"message": "System not initialized."}), 400
        if eye_tracker.is_running():
            return jsonify({"message": "Eye tracking is already running."}), 400

        eye_tracker.start_process()
        return jsonify({"message": "Eye tracking started!"}), 200

    @app.route("/stop", methods=["POST"])
    def stop() -> Dict[str, Any]:
        global eye_tracker
        if eye_tracker is None:
            return jsonify({"message": "System not initialized."}), 400
        if not eye_tracker.is_running():
            return jsonify({"message": "Eye tracking is not running."}), 400

        eye_tracker.stop_process()
        eye_tracker = None
        return jsonify({"message": "Eye tracking stopped!"}), 200

    @app.route("/analyze", methods=["POST"])
    def analyze() -> Dict[str, Any]:
        global mgr
        if mgr.session is not None and len(mgr.view_list) > 0:
            mgr.split_data()
            analyzer = DataAnalyzer(SessionManager.RESOLUTION)
            for i, view in enumerate(mgr.view_list):
                analyzer.extract(view)
                analyzer.draw(view)
                # analyzer.calculate(view)
                print(f"View {i+1} analysis done.")
            return jsonify({"message": "Analysis finished!"}), 200
        else:
            return (
                jsonify(
                    {
                        "message": f"System not initialized. Session={mgr.session}, view_list({len(mgr.view_list)})"
                    }
                ),
                400,
            )
