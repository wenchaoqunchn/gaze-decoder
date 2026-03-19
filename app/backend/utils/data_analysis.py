import csv
import json
from typing import List, Tuple
import numpy as np
from .session_tools import View
from lib import detectors
from lib import gazeplotter as gp


class DataAnalyzer:
    def __init__(self, resolution, fix_threshold=None, sac_threshold=None):
        self.resolution = resolution
        self.f_thresh = (
            fix_threshold
            if fix_threshold is not None
            else {"maxdist": 25, "mindur": 50}
        )
        self.s_thresh = (
            sac_threshold
            if sac_threshold is not None
            else {"minlen": 5, "maxvel": 40, "maxacc": 340}
        )
        self.fixlist = []
        self.saclist = []
        self.xlist = []
        self.ylist = []
        self.nodata = False

    def extract(self, view: View):
        """
        Parse the split-data CSV for a single view and extract fixations and saccades.

        Args:
            view (View): View object whose ``path["split"]`` points to the CSV file.

        Side-effects:
            Sets ``self.fixlist``, ``self.saclist``, ``self.xlist``, ``self.ylist``,
            and ``self.nodata`` based on the file contents.
        """
        filepath = view.path["split"]
        maxdist = self.f_thresh["maxdist"]
        mindur = self.f_thresh["mindur"]
        minlen = self.s_thresh["minlen"]
        maxvel = self.s_thresh["maxvel"]
        maxacc = self.s_thresh["maxacc"]

        with open(filepath, "r") as file:
            reader = csv.reader(file)
            next(reader)  # skip header row
            data = list(reader)
        if data:
            self.nodata = False
            data_array = np.array(data)

            # extract time, x, y columns as float arrays
            tlist = data_array[:, 0].astype(float)
            xlist = data_array[:, 1].astype(float)
            ylist = data_array[:, 2].astype(float)

            # detect fixations and saccades using PyGazeAnalyser detectors
            _, fixlist = detectors.fixation_detection(
                xlist, ylist, tlist, maxdist, mindur
            )
            _, saclist = detectors.saccade_detection(
                xlist, ylist, tlist, minlen, maxvel, maxacc
            )

            self.fixlist = fixlist
            self.saclist = saclist
            self.xlist = xlist
            self.ylist = ylist
            print("Finish calculating.")
        else:
            self.nodata = True
            print("No data to be extracted.")

    def draw(self, view: View):
        if not self.nodata:
            res = self.resolution

            ori = view.path["origin"]
            fix = view.path["fixation"]
            raw = view.path["rawpoint"]
            scan = view.path["scanpath"]
            heat = view.path["heatmap"]

            fixl = self.fixlist
            sacl = self.saclist
            xl = self.xlist
            yl = self.ylist

            gp.draw_fixations(fixl, res, ori, fix)
            gp.draw_heatmap(fixl, res, ori, heat)
            gp.draw_raw(xl, yl, res, ori, raw)
            gp.draw_scanpath(fixl, sacl, res, ori, 0.5, scan)
            print("Finish drawing.")
        else:
            print("No data to be drawn.")

    def calculate(self, view: View):
        self.load_aoi_data("data/AOI.json", view.name)
        self.assign_aoi_to_fixations(self.fixlist)
        self.assign_aoi_to_saccades(self.saclist)
        m = self.calculate_metrics(self.fixlist, self.saclist, self.aoi_path)

        print(m)

    def load_aoi_data(self, aoi_json_path, view_name):
        with open(aoi_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        view_aoi = data.get(view_name)
        self.aoi_list = view_aoi.get("AOIs")
        self.aoi_path = view_aoi.get("path")

    def assign_aoi_to_fixations(self, Efix):
        for fixation in Efix:
            endx, endy = fixation[3], fixation[4]
            for aoi in self.aoi_list:
                pos = aoi["pos"]
                if (pos["x1"] <= endx <= pos["x2"]) and (
                    pos["y1"] <= endy <= pos["y2"]
                ):
                    fixation.append(aoi["id"])  # tag fixation with matching AOI id
                    break

    def assign_aoi_to_saccades(self, Esac):
        for saccade in Esac:
            startx, starty = saccade[3], saccade[4]
            endx, endy = saccade[5], saccade[6]

            start_aoi, end_aoi = None, None
            for aoi in self.aoi_list:
                pos = aoi["pos"]
                if (pos["x1"] <= startx <= pos["x2"]) and (
                    pos["y1"] <= starty <= pos["y2"]
                ):
                    start_aoi = aoi["id"]
                if (pos["x1"] <= endx <= pos["x2"]) and (
                    pos["y1"] <= endy <= pos["y2"]
                ):
                    end_aoi = aoi["id"]

            saccade.append(start_aoi)  # origin AOI id
            saccade.append(end_aoi)  # destination AOI id

    def calculate_metrics(self, Efix, Esac, task_aoi_sequence):
        metrics = {}

        # --- fixation metrics ---
        for aoi in task_aoi_sequence:
            aoi_fixations = [f for f in Efix if f[-1] == aoi]

            metrics[aoi] = {
                "fixation_count": len(aoi_fixations),
                "total_fixation_duration": sum(f[2] for f in aoi_fixations),
                "mean_fixation_duration": (
                    (sum(f[2] for f in aoi_fixations) / len(aoi_fixations))
                    if aoi_fixations
                    else 0
                ),
            }

        # --- saccade metrics ---
        for aoi in task_aoi_sequence:
            aoi_saccades = [
                s for s in Esac if s[-2] == aoi
            ]  # filter by destination AOI

            metrics.setdefault(aoi, {}).update(
                {
                    "saccade_count": len(aoi_saccades),
                    "total_saccade_duration": sum(s[2] for s in aoi_saccades),
                }
            )

        return metrics
