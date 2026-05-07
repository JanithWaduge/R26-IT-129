# Environment Fix Summary

**Date**: May 2, 2026  
**Issue**: Terminal error when running `slsl_server.py`  
**Root Cause**: Incompatible JAX/Jaxlib versions with NumPy 1.24.3

---

## Error Log (Before Fix)

```
AttributeError: module 'numpy' has no attribute 'dtypes'. Did you mean: 'dtype'?
  File "jax/_src/dtypes.py", line 518, in <module>
    if hasattr(np.dtypes, 'StringDType'):
```

### Import Chain
```
import mediapipe
  → tensorflow import
    → tensorflow.lite.python.util imports jax
      → jax._src.dtypes tries np.dtypes (NumPy 2.0+ feature)
        ❌ NumPy 1.24.3 doesn't have dtypes
```

---

## Solution Applied

### 1. **Removed Incompatible Packages**
- Uninstalled `jax==0.6.2` and `jaxlib==0.6.2`
- These were indirect dependencies pulled in by TensorFlow's TFLite converter
- MediaPipe and model inference do not require JAX

**Command:**
```bash
pip uninstall -y jax jaxlib
```

### 2. **Fixed Model Path Resolution** 
- Updated [slsl_server.py](Janith/slsl_server.py) to check multiple model paths:
  - ✅ Local repo: `./models/slsl_model.tflite`
  - ✅ Legacy desktop: `C:\Users\Janith\Desktop\R26-IT-129\Janith\models\slsl_model.tflite`
  - ❌ FileNotFoundError if neither exists

### 3. **Pinned Working Versions**
Generated [requirements.txt](Janith/requirements.txt) with all tested packages.

---

## Working Package Versions

| Package | Version | Notes |
|---------|---------|-------|
| TensorFlow | 2.13.0 | Core ML framework |
| NumPy | 1.24.3 | 2.0 is incompatible with TF 2.13 |
| MediaPipe | 0.10.5 | Hand keypoint extraction |
| Keras | 2.13.1 | High-level model API |
| OpenCV | 4.8.1.78 | Image processing |
| Flask | 3.1.3 | REST API server |

---

## Test Result ✅

**Server now runs successfully:**

```
Loading TFLite model...
INFO: Created TensorFlow Lite delegate for select TF ops.
✅ Model loaded
   Input shape : [ 1 30 63]
   Output shape: [ 1 30]
✅ MediaPipe ready

 * Running on http://172.20.10.5:5000
 * Running on http://127.0.0.1:5000
```

**Endpoints tested:**
- `GET /health` → 200 OK
- `POST /predict_frame` → 200 OK (30 frames processed)
- `POST /predict_sequence?filter=true` → ✅ [B-Filtered] Continuing (68.0%)

---

## How to Use

### Restore Working Environment
```bash
pip install -r Janith/requirements.txt
```

### Run Server
```bash
cd Janith
python slsl_server.py
```

### From Clean Virtual Environment
```bash
python -m venv slsl_env
./slsl_env/Scripts/activate  # Windows
pip install -r requirements.txt
python slsl_server.py
```

---

## Key Takeaways

1. **NumPy version matters**: TensorFlow 2.13.x needs NumPy < 2.0
2. **JAX incompatibility**: JAX 0.6.2 requires NumPy >= 2.0 — remove it if you're using TF 2.13
3. **Transitive dependencies**: Not all indirect imports are needed (JAX isn't used by our code)
4. **Path resolution**: Always fallback to relative paths for model loading in repos

---

## Next Steps (Optional)

- [ ] Upgrade to TensorFlow 2.14+ if NumPy 2.x support is needed
- [ ] Consider using Docker/poetry for isolated reproducible builds
- [ ] Add `requirements.txt` to CI/CD for automated testing
