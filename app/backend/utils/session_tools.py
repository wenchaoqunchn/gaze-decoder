import csv
import re
import time
from os import makedirs, listdir
from os.path import join, exists, isdir
from typing import Dict, List, Tuple
import win32gui, win32ui, win32con, win32print
import pandas as pd
from pathlib import Path


class Helper:
    # @staticmethod
    # def split(raw_path: Path, split_path: Path, save_dir: Path) -> int:
    #     raw_df = pd.read_csv(raw_path)
    #     split_series = pd.read_csv(split_path)["time"]
    #     split_cnt = len(split_series)
    #     start_index = 0

    #     for i in range(split_cnt):
    #         print(f"Split data segment {i+1}.")
    #         if i < split_cnt:
    #             end_index = raw_df[raw_df["time"] >= split_series[i]].index[0]
    #         else:
    #             end_index = raw_df.index[-1] + 1

    #         sub_df = raw_df.iloc[start_index:end_index]

    #         # 保存为 CSV 文件
    #         split_path = save_paths[i]
    #         sub_df.to_csv(split_path, index=False)

    #         # 更新起始索引
    #         start_index = end_index
    #     print("Split finished.")
    #     return split_cnt
    @staticmethod
    def split(raw_path, view_path, save_dir):
        raw_df = pd.read_csv(raw_path)
        # read view_switch CSV to get split timestamps
        view_df = pd.read_csv(view_path)
        print(view_path)
        print(view_df)
        print("Columns in view_df:", view_df.columns.tolist())
        # use the 'time' column as split boundaries
        split_series = view_df["time"]

        # initialise pointers
        raw_index = 0
        split_index = 0
        slice_number = 1

        while split_index < len(split_series):
            # timestamp of the current slice boundary
            current_split_time = split_series.iloc[split_index]
            slice_data = []

            # collect raw rows that fall before the current boundary
            while (
                raw_index < len(raw_df)
                and raw_df["time"].iloc[raw_index] < current_split_time
            ):
                slice_data.append(raw_df.iloc[raw_index])
                raw_index += 1

            # save non-empty slice (or empty placeholder) to file
            if slice_data:
                slice_df = pd.DataFrame(slice_data)
                output_file = save_dir / f"split_data_{slice_number:02}.csv"
                slice_df.to_csv(output_file, index=False)
                print(f"Saved slice {slice_number} to {output_file}")
            else:
                empty_df = pd.DataFrame(columns=raw_df.columns)
                output_file = save_dir / f"split_data_{slice_number:02}.csv"
                empty_df.to_csv(output_file, index=False)
                print(f"Saved empty slice {slice_number} to {output_file}")

            slice_number += 1

            # advance the split-boundary pointer
            split_index += 1

        if raw_index < len(raw_df):
            slice_data = raw_df.iloc[raw_index:]
            if not slice_data.empty:
                output_file = save_dir / f"split_data_{slice_number:02}.csv"
                slice_df = pd.DataFrame(slice_data)
                print(f"Columns in remaining data: {slice_df.columns.tolist()}")
                slice_df.to_csv(output_file, index=False)
                print(f"Saved remaining data to {output_file}")

        print("Splitting complete.")

    @staticmethod
    def get_resolution() -> Tuple[int, int]:
        hDC = win32gui.GetDC(0)
        # horizontal resolution
        w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
        # vertical resolution
        h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
        return (w, h)

    @staticmethod
    def get_max_index(parent_dir: Path) -> int:
        # list all entries in the parent directory
        items = listdir(parent_dir)

        # regex for session folder names of the form session_XX
        session_pattern = re.compile(r"session_(\d{2})", re.IGNORECASE)
        max_number = 0

        # walk entries and keep track of the highest session number
        for item in items:
            if isdir(join(parent_dir, item)):
                match = session_pattern.search(item)
                if match:
                    session_number = int(match.group(1))
                    max_number = max(max_number, session_number)

        return max_number

    @staticmethod
    def save_screenshot(save_path_str: str, resolution: Tuple[int, int]) -> None:
        hwnd = 0
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()  # 修正这里
        saveBitMap = win32ui.CreateBitmap()  # 修正这里
        w, h = resolution
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)  # 修正这里
        saveDC.SelectObject(saveBitMap)
        saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
        saveBitMap.SaveBitmapFile(saveDC, save_path_str)

    # @staticmethod
    # def update_split_time(save_path: Path) -> None:
    #     # 数据切片时间戳文件
    #     with open(save_path, mode="a") as file:
    #         file.write(f"{int(time.time() * 1000)}\n")

    @staticmethod
    def update_view_sequence(save_path: Path, current_page: str) -> None:
        with open(save_path, mode="a") as file:
            file.write(f"{current_page},{int(time.time() * 1000)}\n")

    @staticmethod
    def check_img_cnt(folder_path: Path) -> int:
        pattern = re.compile(r"origin_(\d{2})\.jpg")  # matches origin_XX.jpg filenames
        file_count = 0
        max_number = 0

        for filename in listdir(folder_path):
            match = pattern.match(filename)
            if match:
                # extract the numeric index
                number = int(match.group(1))
                file_count += 1
                max_number = max(max_number, number)

        # verify indices are contiguous (no gaps)
        if file_count != max_number:
            return -1

        return file_count


class Session:
    def __init__(self, parent_dir: Path, index: int, new_session: bool):
        self.parent_dir = Path(parent_dir)
        self.index = index
        self.session_dir = self.parent_dir / f"session_{index:02}"

        self.dir = {}
        self.path = {}
        self.pre = {}

        self.concat_path()

        if new_session:
            self.create_files()

    def concat_path(self):
        """Build all directory and file paths for this session."""
        self.dir["session"] = self.session_dir
        self.dir["img"] = self.session_dir / "img"
        self.dir["split"] = self.session_dir / "split_data"

        self.path["raw"] = self.session_dir / "raw_data.csv"
        # self.path["split"] = self.session_dir / "split_time.csv"
        self.path["view"] = self.session_dir / "view_switch.csv"

        for img_type in SessionManager.IMG_TYPES:
            self.pre[img_type] = str(self.dir["img"] / f"{img_type}_")

        self.pre["split"] = str(self.dir["split"] / "split_data_")

    def create_files(self):
        """Create the required directories and initialise empty CSV files."""
        self.session_dir.mkdir()
        self.dir["img"].mkdir()
        self.dir["split"].mkdir()

        with open(self.path["raw"], "w") as f:
            f.write("time,x,y\n")

        # with open(self.path["split"], "w") as f:
        #     f.write("view,time\n")  # write header

        with open(self.path["view"], "w") as f:
            f.write("view,time\n")  # write header


class View:

    def __init__(self, index: int, name: str, prefix: Dict[str, str]):
        self.index = index
        self.name = name
        self.path = {}
        """Generate file paths for each image type and the split-data CSV."""
        for t in SessionManager.IMG_TYPES:
            self.path[t] = Path(prefix[t] + f"{self.index:02}.jpg")
        self.path["split"] = Path(prefix["split"] + f"{self.index:02}.csv")


class SessionManager:
    IMG_TYPES = ("origin", "fixation", "rawpoint", "scanpath", "heatmap")
    # RESOLUTION = Helper.get_resolution()
    RESOLUTION = (2560, 1600)
    # VIEWS = [
    #     "News",
    #     "Agenda",
    #     "More",
    #     "Library",
    #     "More2",
    #     "SelectSpace",
    #     "SelectDateEaseOfUse",
    #     "Confirm",
    #     "SelectSeatDeviceEfficiency",
    # ]

    def __init__(self, parent_dir=Path("data")):
        self.parent_dir = parent_dir
        self.session = None
        self.focus_session = 0
        self.new_session = True
        self.view_list = []
        self.switch_cnt = 0

    def init_session(self, focus_session=0):

        self.focus_session = focus_session
        self.new_session = self.focus_session == 0
        self.view_list = []
        self.switch_cnt = 0
        p_dir = self.parent_dir
        max_index = Helper.get_max_index(p_dir)
        if self.new_session:
            if not exists(p_dir):
                makedirs(p_dir)
            s_index = max_index + 1
            self.session = Session(p_dir, s_index, self.new_session)
        else:
            if not exists(p_dir):
                raise FileNotFoundError(f"Folder {p_dir} not found.")
            elif focus_session < 1 or focus_session > max_index:
                raise FileNotFoundError(f"session_{focus_session:02} not found.")

            s_index = self.focus_session
            self.session = Session(p_dir, s_index, self.new_session)

            with open(self.session.path["view"], mode="r", encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader)  # skip header row

                # store view names in order
                view_name = [row[0] for row in reader]  # first column = view name

            view_num_1 = len(pd.read_csv(self.session.path["view"]))
            view_num_2 = Helper.check_img_cnt(self.session.dir["img"])
            if view_num_1 == view_num_2:
                for i in range(view_num_1):
                    self.view_list.append(View(i + 1, view_name[i], self.session.pre))
            else:
                raise RuntimeError("Broken files.")

    def switch_view(self, current_page: str):
        if self.session is None:
            raise RuntimeError("Session not initialized.")
        elif not self.new_session:
            raise RuntimeError("Switching unavailable in focus session mode.")
        else:
            self.switch_cnt += 1
            index = self.switch_cnt
            img_type = SessionManager.IMG_TYPES[0]
            origin_path = Path(self.session.pre[img_type] + f"{index:02}.jpg")
            # split_path = Path(self.session.path["split"])
            view_path = Path(self.session.path["view"])
            res = SessionManager.RESOLUTION

            # Helper.update_split_time(save_path=split_path)
            Helper.update_view_sequence(save_path=view_path, current_page=current_page)
            Helper.save_screenshot(save_path_str=str(origin_path), resolution=res)

            self.view_list.append(
                View(index=index, name=current_page, prefix=self.session.pre)
            )

    def split_data(self):
        if self.session is None:
            raise RuntimeError("Session not initialized.")
        else:
            raw_path = self.session.path["raw"]
            view_path = self.session.path["view"]
            # save_paths = [v.path["split"] for v in self.view_list]
            save_dir = self.session.dir["split"]

            Helper.split(
                raw_path=raw_path,
                view_path=view_path,
                save_dir=save_dir,
            )


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.2f}".format)
    mng = SessionManager()
    mng.init_session()
    mng.split_data()
