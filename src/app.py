import subprocess
from pathlib import Path
import threading
import re
import shlex
import os
import logging

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify

OUTPUT_DIR = Path("compare_out")
DEFAULT_DATA = "data/carla.yaml"
DEFAULT_DET_WEIGHTS = "weights/yolov3_9_5.pt"
DEFAULT_NSR_WEIGHTS = "NRP_weights/NRP_weight.pth"
DEFAULT_TEXTURE_SIZE = 6
DEFAULT_NSR_WEIGHTS = "NRP_weights/NRP_weight.pth"
DEFAULT_TEXTURE_SIZE = 6
DEFAULT_IMG_SIZE = 640

# Add this utility
def rmtree_silent(path: Path):
    import shutil
    try:
        if path.exists():
            shutil.rmtree(path)
    except:
        pass


def find_pairs(output_dir: Path):
    pairs = []
    for orig_path in output_dir.glob("*_orig_det.jpg"):
        base = orig_path.name.replace("_orig_det.jpg", "")
        camo_path = orig_path.with_name(f"{base}_camo_det.jpg")
        if camo_path.exists():
            pairs.append(
                {
                    "id": base,
                    "orig": orig_path.as_posix(),
                    "camo": camo_path.as_posix(),
                }
            )
    return sorted(pairs, key=lambda x: x["id"])


def get_available_textures():
    """Recursively find texture.npy files in the textures directory."""
    # User specified 'src/texture', but fs indicates 'src/textures'. Checking both/using existing.
    # Assuming CWD is src/
    search_paths = [Path("textures"), Path("texture")]
    textures = []
    
    for p in search_paths:
        if p.exists():
            textures.extend([f.as_posix() for f in p.glob("**/*.npy")])
            
    return sorted(list(set(textures)))



def clear_directory(path: Path):
    """Clear all files in the given directory."""
    if not path.exists():
        return
    for item in path.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item)
        except Exception as e:
            print(f"Failed to delete {item}: {e}")

# Selective Log Filtering
class AccessLogFilter(logging.Filter):
    def filter(self, record):
        # Filter out /training_status and /ablation_status requests
        msg = record.getMessage()
        return "GET /training_status" not in msg and "GET /ablation_status" not in msg

# Apply filter to Werkzeug logger
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO) 
log.addFilter(AccessLogFilter())

# Global State
TRAINING_STATE = {
    "status": "idle",
    "progress": 0,
    "batch_progress": 0,
    "current_epoch": 0,
    "total_epochs": 10,
    "log": ["Waiting to start..."],
    "config": {}
}
CURRENT_PROCESS = None

ABLATION_STATE = {
    "status": "idle", # idle, running, completed, error
    "active_cid": None, # 'full', 'no_det', 'no_smooth'
    "active_name": "", # '完整方法...'
    "epoch_current": 0,
    "epoch_total": 0,
    "progress_pct": 0, # Percentage for the active bar
    "results": [],
    "log": [], # For status text
    "error": None
}

def ablation_worker(configs_selected, datapath, epochs, data_fraction, device="0"):
    global ABLATION_STATE
    import json, random, shutil, time, yaml

    # 初始化状态
    ABLATION_STATE["status"] = "running"
    ABLATION_STATE["results"] = []
    ABLATION_STATE["comparisons"] = []
    ABLATION_STATE["error"] = None

    print(f"[DEBUG] Ablation Task Started.")
    print(f"        Training on:   {datapath} (User Input)")
    print(f"        Evaluating on: data/test_dataset (Fixed)")

    config_map = {
        "full": {"name": "完整方法 (Full Method)", "w_det": "1.0", "t": "0.0001"},
        "no_det": {"name": "无检测损失 (No Det Loss)", "w_det": "0.0", "t": "0.0001"},
        "no_smooth": {"name": "无平滑损失 (No Smooth Loss)", "w_det": "1.0", "t": "0.0"},
    }

    # ================= 1. 准备可视化用的临时数据 (复刻目录结构) =================
    temp_source = Path("temp_ablation_source")
    try:
        if temp_source.exists():
            shutil.rmtree(temp_source, ignore_errors=True)
            time.sleep(0.5)
    except: pass

    # 确定源数据根目录
    if Path("data/test_dataset").exists():
        vis_source_root = Path("data/test_dataset")
    else:
        vis_source_root = Path(datapath) if datapath else Path("data/dataset")

    print(f"[DEBUG] Sampling 4 random images from: {vis_source_root.as_posix()}")

    # 扫描源图片
    img_source_dir = vis_source_root / "test_new"
    if not img_source_dir.exists():
        all_images = list(vis_source_root.rglob("*.png")) + list(vis_source_root.rglob("*.jpg"))
    else:
        all_images = list(img_source_dir.glob("*.png")) + list(img_source_dir.glob("*.jpg"))

    candidates = [p for p in all_images if "mask" not in p.name.lower()]
    random.shuffle(candidates)

    selected_samples = []

    # 重建标准目录结构
    target_dirs = {
        "img": temp_source / "test_new",
        "mask": temp_source / "masks",
        "npz": temp_source / "test",
        "label": temp_source / "test_label_new"
    }
    for d in target_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    count_needed = 4
    for img_path in candidates:
        if len(selected_samples) >= count_needed: break

        stem = img_path.stem
        src_img = img_path
        src_mask = vis_source_root / "masks" / img_path.name
        if not src_mask.exists(): src_mask = vis_source_root / "masks" / (stem + ".png")
        src_npz = vis_source_root / "test" / (stem + ".npz")
        src_label = vis_source_root / "test_label_new" / (stem + ".txt")

        if src_img.exists() and src_mask.exists() and src_npz.exists():
            try:
                shutil.copy(src_img, target_dirs["img"] / src_img.name)
                shutil.copy(src_mask, target_dirs["mask"] / (stem + ".png"))
                shutil.copy(src_npz, target_dirs["npz"] / src_npz.name)
                if src_label.exists():
                    shutil.copy(src_label, target_dirs["label"] / src_label.name)
                else:
                    with open(target_dirs["label"] / (stem + ".txt"), 'w') as f: pass
                selected_samples.append((img_path, src_mask))
            except Exception as e:
                print(f"[WARN] Failed to copy sample {stem}: {e}")

    print(f"[DEBUG] Prepared {len(selected_samples)} valid samples.")

    # 生成临时 YAML
    temp_yaml_path = temp_source / "temp_data.yaml"
    temp_yaml_content = {
        'path': str(temp_source.absolute()),
        'train': 'test_new', 'val': 'test_new', 'test': 'test_new',
        'names': {0: 'car'}
    }
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(temp_yaml_content, f)

    for cache in temp_source.rglob("*.cache"):
        try: cache.unlink()
        except: pass

    # ================= 3. 主循环 =================
    try:
        # 准备原图存放文件夹 (Cleaning Orig folder)
        orig_out_dir = OUTPUT_DIR / "ablation_aligned" / "orig"
        if orig_out_dir.exists(): shutil.rmtree(orig_out_dir)
        orig_out_dir.mkdir(parents=True, exist_ok=True)

        for cid in configs_selected:
            if cid not in config_map: continue
            cfg = config_map[cid]

            ABLATION_STATE["active_cid"] = cid
            ABLATION_STATE["active_name"] = cfg["name"]
            ABLATION_STATE["epoch_current"] = 0
            ABLATION_STATE["epoch_total"] = int(epochs)
            ABLATION_STATE["progress_pct"] = 0

            # --- A. 训练 ---
            train_yaml = "data/carla.yaml"
            if Path("data/carla.yaml").exists(): pass

            ABLATION_STATE["log"] = [f"正在训练 {cfg['name']}..."]
            print(f"[INFO] Training {cid}...")

            train_cmd = [
                "python", "train_camouflage.py",
                "--datapath", datapath,
                "--data", train_yaml,
                "--epochs", epochs,
                "--w_det", cfg["w_det"],
                "--t", cfg["t"],
                "--data_fraction", str(float(data_fraction)/100.0),
                "--batch-size", "1",
                "--device", str(device),
                "--name", f"ablation_{cid}"
            ]

            process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=os.getcwd())
            parser = LogParser()
            run_state = { "current_epoch": 0, "total_epochs": int(epochs), "progress": 0 }
            for line in iter(process.stdout.readline, ''):
                if not line: break
                _, run_state = parser.parse_line(line, run_state)
                ABLATION_STATE["epoch_current"] = run_state.get("current_epoch", 0)
                ABLATION_STATE["epoch_total"] = run_state.get("total_epochs", int(epochs))
                ABLATION_STATE["progress_pct"] = run_state.get("progress", 0)
                ABLATION_STATE["log"] = [f"正在训练 {cfg['name']} - Epoch {ABLATION_STATE['epoch_current']}/{ABLATION_STATE['epoch_total']}"]
            process.wait()

            expected_texture = f"textures/texture_ablation_{cid}.npy"
            if not os.path.exists(expected_texture):
                print(f"[ERROR] Texture missing for {cid}, skipping.")
                continue

            # --- B. 全量评估 (临时文件夹，用完即删) ---
            TEST_SET_PATH = "data/test_dataset"
            TEST_SET_YAML = "data/carla_test.yaml"
            if not Path(TEST_SET_PATH).exists(): continue

            ABLATION_STATE["log"] = [f"正在评估 {cfg['name']}..."]

            # 使用临时文件夹存放 metrics.json
            temp_eval_dir = OUTPUT_DIR / f"temp_metrics_{cid}"
            temp_eval_dir.mkdir(parents=True, exist_ok=True)

            gen_cmd_metrics = [
                "python", "gen_camo_detect.py",
                "--textures", expected_texture,
                "--datapath", TEST_SET_PATH,
                "--data", TEST_SET_YAML,
                "--output", temp_eval_dir.as_posix(),
                "--conf_thres", "0.5",
                "--device", str(device),
                "--no_save",      # 不存图
                "--workers", "0"
            ]

            subprocess.run(gen_cmd_metrics, check=False)

            metrics = {"asr": 0, "avg_conf_camo": 0}
            metrics_file = temp_eval_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)

            # === 清理临时评估文件夹 (保持目录整洁) ===
            try:
                shutil.rmtree(temp_eval_dir)
            except: pass

            ABLATION_STATE["results"].append({
                "name": cfg["name"],
                "metrics": metrics,
                "dir": f"ablation_{cid}"
            })
            with open("ablation_results.json", "w") as f:
                json.dump(ABLATION_STATE["results"], f, indent=2)


            # --- C. 可视化生成 (分离存储) ---
            ABLATION_STATE["log"] = [f"正在生成可视化对比图 {cfg['name']}..."]

            # 1. 生成到 cid 文件夹
            vis_out_dir = OUTPUT_DIR / "ablation_aligned" / cid
            if vis_out_dir.exists(): shutil.rmtree(vis_out_dir)
            vis_out_dir.mkdir(parents=True, exist_ok=True)

            gen_cmd_vis = [
                "python", "gen_camo_detect.py",
                "--textures", expected_texture,
                "--datapath", temp_source.as_posix(),
                "--data", temp_yaml_path.as_posix(),
                "--output", vis_out_dir.as_posix(),
                "--conf_thres", "0.5",
                "--device", str(device),
                "--max_samples", str(len(selected_samples)),
                "--workers", "0"
            ]

            vis_proc = subprocess.run(gen_cmd_vis, check=False)

            # 2. 搬运原图到 orig 文件夹
            # 脚本会生成 {base}_orig_det.jpg 和 {base}_camo_det.jpg
            if vis_proc.returncode == 0:
                for jpg_file in vis_out_dir.glob("*_orig_det.jpg"):
                    try:
                        # 移动文件到 orig 文件夹 (如果已存在则覆盖)
                        shutil.move(str(jpg_file), str(orig_out_dir / jpg_file.name))
                    except Exception as mv_e:
                        print(f"[WARN] Failed to move {jpg_file.name}: {mv_e}")


        # ================= 4. 整理最终对比数据 =================
        print("[DEBUG] Building comparison JSON...")
        comparison_data = []
        sample_bases = [p.stem for (p, m) in selected_samples]

        # 此时 orig 文件夹里应该有原图，cid 文件夹里只有伪装图

        for base in sample_bases:
            # 原图路径指向 orig 文件夹
            orig_rel_path = f"ablation_aligned/orig/{base}_orig_det.jpg"

            entry = {
                "id": base,
                "orig": orig_rel_path,
                "variants": []
            }

            if not (OUTPUT_DIR / orig_rel_path).exists():
                continue

            for cid in configs_selected:
                # 伪装图还在各自的文件夹里
                camo_rel_path = f"ablation_aligned/{cid}/{base}_camo_det.jpg"
                if (OUTPUT_DIR / camo_rel_path).exists():
                    entry["variants"].append({
                        "cid": cid,
                        "name": config_map[cid]["name"],
                        "path": camo_rel_path
                    })
            comparison_data.append(entry)

        ABLATION_STATE["comparisons"] = comparison_data
        with open("ablation_comparisons.json", "w") as f:
            json.dump(comparison_data, f, indent=2)

        print(f"[INFO] Experiment Complete. Generated {len(comparison_data)} comparison sets.")
        ABLATION_STATE["status"] = "completed"
        ABLATION_STATE["log"] = ["实验全部完成"]

    except Exception as e:
        import traceback
        traceback.print_exc()
        ABLATION_STATE["status"] = "error"
        ABLATION_STATE["error"] = str(e)
        ABLATION_STATE["log"] = [f"Error: {str(e)}"]


class LogParser:
    def __init__(self):
        # Regex Patterns
        self.epoch_pattern = re.compile(r"(\d+)/(\d+)\s+[0-9.]+[GM]")
        self.batch_pattern = re.compile(r"(\d+)%\|")
        self.batch_count_pattern = re.compile(r"\s(\d+)/(\d+)\s+\[")
        self.time_duration_pattern = re.compile(r"\[([0-9:]+)<([0-9:]+)")
        self.loss_pattern = re.compile(r"(?:Loss|loss)[:=]\s*([0-9.]+)")
        self.time_pattern = re.compile(r"([0-9.]+)s/it")
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def parse_line(self, line, state):
        """Parses a single log line and updates the state dict."""
        clean_line = self.ansi_escape.sub('', line).strip()
        if not clean_line:
            return clean_line, state

        # Parse Epoch and Detailed Losses (Table format)
        match = self.epoch_pattern.search(clean_line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            state["current_epoch"] = current
            state["total_epochs"] = total
            
            parts = clean_line.split()
            # parts: 0=Epoch/Total, 1=Mem, 2=TotalLoss(a_ls), 3=SmoothLoss(s_ls), 4=DetLoss(t_loss)
            if len(parts) >= 5:
                    try:
                        val_total = float(parts[2])
                        val_smooth = float(parts[3])
                        val_det = float(parts[4])
                        state["current_loss"] = val_total
                        state["loss_smooth"] = val_smooth
                        state["loss_det"] = val_det
                    except:
                        pass

            # Major progress calculation
            major_base = ((current - 1) / total) * 100
            batch_contrib = 0
            if "batch_progress" in state:
                batch_contrib = (state["batch_progress"] / 100) * (100 / total) if total > 0 else 0
            state["progress"] = int(major_base + batch_contrib)

        # Parse Batch Progress (Percentage)
        batch_match = self.batch_pattern.search(clean_line)
        if batch_match and state.get("current_epoch", 0) > 0:
            b_prog = int(batch_match.group(1))
            state["batch_progress"] = b_prog
            
            # Check for explicit batch counts X/Y
            count_match = self.batch_count_pattern.search(clean_line)
            if count_match:
                state["batch_current"] = int(count_match.group(1))
                state["batch_total"] = int(count_match.group(2))
            
            # Check for Time Duration [Elapsed<Remaining]
            dur_match = self.time_duration_pattern.search(clean_line)
            if dur_match:
                state["time_elapsed"] = dur_match.group(1)
                state["time_remaining"] = dur_match.group(2)

            # Update total progress calculation
            c = state["current_epoch"]
            t = state["total_epochs"]
            if t > 0 and c > 0:
                    major_base = ((c - 1) / t) * 100
                    batch_contrib = (b_prog / 100) * (100 / t)
                    state["progress"] = int(major_base + batch_contrib)

        # Parse Loss (Legacy/Fallback)
        if "Loss:" in clean_line:
            loss_match = self.loss_pattern.search(clean_line)
            if loss_match:
                state["current_loss"] = float(loss_match.group(1))

        # Parse Rate (s/it)
        time_match = self.time_pattern.search(clean_line)
        if time_match:
            state["epoch_time"] = time_match.group(1) + "s/it"
            
        return clean_line, state

def monitor_training(cmd, config):
    """Runs the training command in a thread and monitors output."""
    global TRAINING_STATE, CURRENT_PROCESS
    TRAINING_STATE["status"] = "running"
    TRAINING_STATE["progress"] = 0
    TRAINING_STATE["log"] = ["Training started..."]
    TRAINING_STATE["config"] = config
    
    parser = LogParser()
    
    # Initialize total_epochs immediately from config to show correct target
    try:
        if config and "epochs" in config:
            TRAINING_STATE["total_epochs"] = int(config["epochs"])
            TRAINING_STATE["current_epoch"] = 0
            # Reset progress/logs on new run
            TRAINING_STATE["progress"] = 0
            TRAINING_STATE["batch_progress"] = 0
            TRAINING_STATE["batch_current"] = 0
            TRAINING_STATE["batch_total"] = 0 
    except:
        pass
    
    try:
        # bufsize 1 means line buffered
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1,
            cwd=os.getcwd() # Ensure correct working directory
        )
        CURRENT_PROCESS = process
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            clean_line, TRAINING_STATE = parser.parse_line(line, TRAINING_STATE)
            
            if clean_line:
                 # Keep log
                if len(TRAINING_STATE["log"]) > 50:
                    TRAINING_STATE["log"].pop(0)
                TRAINING_STATE["log"].append(clean_line)
                
        process.wait()
        
        # Check if it was terminated manually or finished
        if TRAINING_STATE["status"] == "terminated":
            TRAINING_STATE["log"].append("Process was terminated by user.")
        elif process.returncode == 0:
            TRAINING_STATE["status"] = "completed"
            TRAINING_STATE["progress"] = 100
            TRAINING_STATE["batch_progress"] = 100
            TRAINING_STATE["current_epoch"] = TRAINING_STATE["total_epochs"]
            TRAINING_STATE["log"].append("Training completed successfully.")
        else:
            TRAINING_STATE["status"] = "error"
            TRAINING_STATE["log"].append(f"Training failed with return code {process.returncode}")
            
    except Exception as e:
        TRAINING_STATE["status"] = "error"
        TRAINING_STATE["log"].append(f"Error during execution: {str(e)}")
    finally:
        CURRENT_PROCESS = None


def create_app():
    app = Flask(__name__)
    app.secret_key = "dev"  # simple dev secret, change for production
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @app.route("/", methods=["GET"])
    def home():
        return render_template("home.html")

    @app.route("/results", methods=["GET"])
    def results():
        pairs = find_pairs(OUTPUT_DIR)
        textures = get_available_textures()
        return render_template(
            "results.html",
            pairs=pairs,
            textures=textures,
            defaults={
                "data": DEFAULT_DATA,
                "det_weights": DEFAULT_DET_WEIGHTS,
                "nsr_weights": DEFAULT_NSR_WEIGHTS,
                "texture_size": DEFAULT_TEXTURE_SIZE,
                "img_size": DEFAULT_IMG_SIZE,
            },
        )

    @app.route("/training", methods=["GET"])
    def training():
        return render_template("training.html")

    @app.route("/compare_out/<path:filename>")
    def serve_image(filename):
        return send_from_directory(OUTPUT_DIR, filename)

    @app.route("/generate", methods=["POST"])
    def generate():
        # Clear output directory before generating
        clear_directory(OUTPUT_DIR)
        
        textures = request.form.get("textures", "").strip()
        datapath = request.form.get("datapath", "").strip()
        max_samples = request.form.get("max_samples", "4").strip()
        data = request.form.get("data", DEFAULT_DATA).strip()
        det_weights = request.form.get("det_weights", DEFAULT_DET_WEIGHTS).strip()
        nsr_weights = request.form.get("nsr_weights", DEFAULT_NSR_WEIGHTS).strip()
        texture_size = request.form.get("texture_size", str(DEFAULT_TEXTURE_SIZE)).strip()
        img_size = request.form.get("img_size", str(DEFAULT_IMG_SIZE)).strip()

        if not textures or not datapath:
            flash("Missing required arguments.", "error")
            return jsonify({"status": "error", "message": "Missing arguments."})

        # Auto-detect config for test_dataset
        if "test_dataset" in datapath and "carla.yaml" in data:
            test_config_path = Path("data/carla_test.yaml")
            if test_config_path.exists():
                data = "data/carla_test.yaml"
                TRAINING_STATE["log"].append("Auto-switched data config to data/carla_test.yaml for test dataset.")

        cmd = [
            "python",
            "gen_camo_detect.py",
            "--textures",
            textures,
            "--datapath",
            datapath,
            "--data",
            data,
            "--det_weights",
            det_weights,
            "--nsr_weights",
            nsr_weights,
            "--output",
            OUTPUT_DIR.as_posix(),
            "--device",
            request.form.get("device", "0").strip() or "0",
            "--max_samples",
            max_samples or "0",
            "--texture_size",
            texture_size,
            "--img_size",
            img_size,
        ]

        try:
            # Command execution
            subprocess.run(cmd, check=True)
            flash("生成完成", "success")
        except subprocess.CalledProcessError as e:
            flash(f"生成失败: {str(e)}", "error")

        return redirect(url_for("results"))


    @app.route('/run_training', methods=['POST'])
    def run_training():
        """Handles training request."""
        if request.method == 'POST':
            if TRAINING_STATE["status"] == "running":
                 flash("Training is already in progress.", "warning")
                 return redirect(url_for('training'))

            # Collect parameters
            datapath = request.form.get("datapath", "").strip()
            data_yaml = request.form.get("data_yaml", "").strip()
            obj_file = request.form.get("obj_file", "").strip()
            epochs = request.form.get("epochs", "10").strip()
            batch_size = request.form.get("batch_size", "1").strip()
            lr = request.form.get("lr", "0.01").strip()
            device = request.form.get("device", "0").strip()
            
            # New Feature: Data Fraction and Sampling
            data_fraction_pct = request.form.get("data_fraction", "100").strip()
            if not data_fraction_pct: data_fraction_pct = "100"
            data_fraction = float(data_fraction_pct) / 100.0
            sampling_method = request.form.get("sampling_method", "random").strip()
            
            cmd = [
                "python", "train_camouflage.py",
                "--datapath", datapath,
                "--data", data_yaml,
                "--obj_file", obj_file,
                "--epochs", epochs,
                "--batch-size", batch_size,
                "--lr", lr,
                "--device", device,
                "--data_fraction", str(data_fraction),
                "--data_sampling", sampling_method
            ]
            
            # Config dict for restoration
            config = {
                "datapath": datapath,
                "data_yaml": data_yaml,
                "obj_file": obj_file,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "device": device,
                "data_fraction": data_fraction_pct,
                "sampling_method": sampling_method
            }
            
            # Start background thread
            thread = threading.Thread(target=monitor_training, args=(cmd, config))
            thread.daemon = True
            thread.start()
            
            return redirect(url_for('training'))
        return redirect(url_for('training'))

    @app.route('/stop_training', methods=['POST'])
    def stop_training():
        """Terminates the current training process."""
        global CURRENT_PROCESS, TRAINING_STATE
        
        # Use local reference to avoid race condition with monitor thread clearing global
        proc = CURRENT_PROCESS
        
        if proc and proc.poll() is None:
            pass 
            
        try:
            if proc and proc.poll() is None:
                proc.terminate() # Send SIGTERM
                try:
                    proc.wait(timeout=5) # Wait for process to terminate
                except subprocess.TimeoutExpired:
                    # If still running after timeout
                    if proc.poll() is None: 
                        proc.kill() # Send SIGKILL
                        proc.wait()
                
                TRAINING_STATE["status"] = "terminated"
                TRAINING_STATE["log"].append("Training process terminated by user.")
                flash("Training stopped.", "info")
                return jsonify({"status": "success", "message": "Training stopped."})
            else:
                flash("No active training process found.", "warning")
                return jsonify({"status": "error", "message": "No active training process found."})
        except Exception as e:
            TRAINING_STATE["status"] = "error"
            TRAINING_STATE["log"].append(f"Error while stopping training: {str(e)}")
            flash(f"Error stopping training: {str(e)}", "error")
            return jsonify({"status": "error", "message": f"Error stopping training: {str(e)}"})
        finally:
            CURRENT_PROCESS = None

    @app.route('/training_status')
    def training_status():
        """Returns the current training state."""
        return jsonify(TRAINING_STATE)

    @app.route("/ablation", methods=["GET"])
    def ablation():
        return render_template("ablation.html")

    @app.route("/run_ablation", methods=["POST"])
    def run_ablation():
        if ABLATION_STATE["status"] == "running":
            return jsonify({"status": "running", "message": "Experiment already running"})
            
        datapath = request.form.get("datapath", "").strip()
        epochs = request.form.get("epochs", "5").strip()
        data_fraction = request.form.get("data_fraction", "10").strip()
        device = request.form.get("device", "0").strip()
        configs_selected = request.form.getlist("configs")

        # Debug info for troubleshooting button/trigger behavior
        print(f"[ABLA] start request :: datapath={datapath}, epochs={epochs}, data_fraction={data_fraction}, device={device}, configs={configs_selected}")
        
        t = threading.Thread(
            target=ablation_worker,
            args=(configs_selected, datapath, epochs, data_fraction, device)
        )
        t.daemon = True
        t.start()
        
        print("[ABLA] run_ablation accepted and background thread started")
        return jsonify({"status": "started"})
        
    @app.route("/ablation_status")
    def ablation_status():
        return jsonify(ABLATION_STATE)

    @app.route("/get_last_ablation_results")
    def get_last_ablation_results():
        """Returns the results AND comparisons of the last successful ablation run."""
        data = {"results": [], "comparisons": []}
        
        # Load Metrics
        results_path = Path("ablation_results.json")
        if results_path.exists():
            import json
            try:
                with open(results_path) as f:
                    data["results"] = json.load(f)
            except:
                pass
        
        # Load Comparisons
        comp_path = Path("ablation_comparisons.json")
        if comp_path.exists():
            import json
            try:
                 with open(comp_path) as f:
                     data["comparisons"] = json.load(f)
            except:
                pass

        # Fallback to current memory if file read fails or is empty
        if ABLATION_STATE["results"]:
            data["results"] = ABLATION_STATE["results"]
        if "comparisons" in ABLATION_STATE and ABLATION_STATE["comparisons"]:
             data["comparisons"] = ABLATION_STATE["comparisons"]
             
        # Debug info for front-end fetch of results
        print(f"[ABLA] get_last_ablation_results called :: results={len(data['results'])} comparisons={len(data['comparisons'])}")
        
        return jsonify(data)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="localhost", port=5000, debug=True)
