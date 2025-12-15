#!/usr/bin/env python
"""
cbf_to_vector.py

Read CBF/imgCIF image file, convert to numpy array, produce embedding vector.

Provides:
    - read_cbf_image(path): -> numpy.ndarray (H, W) as dtype float32
    - image_to_embedding(img, method='resnet', size=224): -> 1D numpy vector
    - cbf_to_vector(path, method='resnet'|'pixel', size=224): -> vector
"""

import os
import uuid
import subprocess
import numpy as np
from PIL import Image
import io
import warnings
import argparse
from glob import glob

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


# try to import available readers
try:
    import dxtbx  # dials/dxtbx (optional, high-level image reader)
    from dxtbx.format.FormatCBF import FormatCBF
    HAVE_DXTBX = True
except Exception:
    print("WARNING: do not have dxtbx")
    HAVE_DXTBX = False

try:
    import pycbf
    HAVE_PYCBF = True
except Exception:
    print("WARNING: do not have pycbf")
    HAVE_PYCBF = False

# deep model (torch)
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as models
    HAVE_TORCH = True
except Exception:
    print("WARNING: do not have torch")
    HAVE_TORCH = False


def read_cbf_with_dxtbx_orig(path):
    """Read using dxtbx if available. Returns numpy 2D array (float32)."""
    if not HAVE_DXTBX:
        raise RuntimeError("dxtbx not installed")
    fmt = FormatCBF(path)
    # FormatCBF provides get_raw_data() / get_raw_data_as_array in some builds
    try:
        arr = fmt.get_raw_data_as_array().astype(np.float32)
    except Exception as exp:
        # fallback to get_raw_data which may return flex arrays
        raw = fmt.get_raw_data()
        try:
            # convert flex array to numpy if present
            arr = np.asarray(raw, dtype=np.float32)
        except Exception:
            raise
    return arr

def read_cbf_with_dxtbx(path):
    if not HAVE_DXTBX:
        raise RuntimeError("dxtbx not installed")
    fmt = FormatCBF(path)
    
    try:
        # Newer dxtbx sometimes has get_raw_data_as_numpy (depending on build)
        arr = fmt.get_raw_data().as_numpy_array().astype(np.float32)
    except AttributeError:
        #print("WARNING: unable to get image via get_raw_data")
        arr = read_cbf_with_fabio(path)
    return arr

def read_cbf_with_fabio(path):
    import fabio
    img = fabio.open(path).data
    return img.astype(np.float32)

def read_cbf_with_pycbf(path):
    """Read using pycbf (CBFlib binding). Returns numpy 2D array (float32)."""
    if not HAVE_PYCBF:
        raise RuntimeError("pycbf not installed")

    # Note: pycbf has a SWIG-derived API. This routine uses common patterns:
    # - load file with cbf_read_widefile into a cbf_handle_struct.
    # - iterate categories/arrays to find binary section and read integerarray.
    # The specific API may vary; check your pycbf docs if something fails.
    handle = pycbf.cbf_handle_struct()
    # cbf_read_widefile(handle, filename) returns 0 on success (CBFlib API)
    rc = pycbf.cbf_read_widefile(handle, path)
    if rc != 0:
        raise RuntimeError(f"pycbf failed to read {path}: rc={rc}")

    # Example approach: find the binary array for the detector image (element 0)
    # Many CBFs have a binary section that maps to an integer array of pixels.
    # The pycbf python wrapper exposes low-level calls; here we attempt to fetch
    # the "array" via convenience properties if available (some builds expose .array)
    try:
        # Attempt to use a high-level convenience method if present
        # (This is not universally available; see pycbf docs)
        arr = pycbf.cbf_get_integerarray(handle)  # may not exist
    except Exception:
        # fallback: try to find a detector/array node and convert manually.
        # Because the pycbf API varies, we'll do a defensive approach:
        # walk categories and rows — try common patterns
        # NOTE: this code may require small edits depending on your pycbf version.
        import ctypes

        # Helper: attempt to read first image block as bytes using cbf_get_integerarray_from_handle
        # Many pycbf builds provide methods similar to CBFlib C API - consult docs.
        raise RuntimeError("pycbf read succeeded but automatic extraction not implemented for this pycbf build. "
                           "Check pycbf docs and adapt read_cbf_with_pycbf() accordingly.")

    # ensure numpy float32
    arr = np.array(arr, dtype=np.float32)
    return arr


def read_cbf_with_convert(path):
    """
    Fallback: try to convert the CBF file to a TIFF/PNG using external tools
    (e.g., cbf2img, cbf_dump, or other). The tool must be available in PATH.
    We attempt 'cbf2tif' -> read TIFF, or 'cbf_dump' -> extract the binary section.
    """
    # try cbf2tif / cbf2img utilities
    tmp_tif = path + ".conv.tif"
    tried = []
    # common helper names
    tools = ["cbf2tiff", "cbf2tif", "cbf2img", "cbf_dump", "cbfdump"]
    for t in tools:
        if shutil.which(t):
            try:
                if t in ("cbf2tiff", "cbf2tif", "cbf2img"):
                    subprocess.check_call([t, path, tmp_tif])
                    img = Image.open(tmp_tif)
                    arr = np.asarray(img).astype(np.float32)
                    try:
                        os.remove(tmp_tif)
                    except Exception:
                        pass
                    return arr
                elif t in ("cbf_dump", "cbfdump"):
                    # cbf_dump may print raw data; try to capture and parse (complex)
                    # For now simply raise to signal fallback complexity.
                    raise RuntimeError(f"{t} found but automatic parsing not implemented.")
            except subprocess.CalledProcessError:
                tried.append(t)
                continue

    raise RuntimeError(f"No suitable external CBF conversion tool found (tried: {tools}). "
                       "Install cbf2tiff/cbf2img or pycbf/dxtbx.")


import shutil

def read_cbf_image(path):
    """Master reader that tries multiple backends and returns numpy 2D float32 image."""
    #print("### read_cbf_image", path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # 1) dxtbx (if present)
    if HAVE_DXTBX:
        try:
            print("reading via dxtbx")
            return read_cbf_with_dxtbx(path)
        except Exception as e:
            warnings.warn(f"dxtbx reader failed: {e}; trying pycbf/fallback")

    # 2) pycbf (if present)
    if HAVE_PYCBF:
        try:
            print("reading via pycbf")
            return read_cbf_with_pycbf(path)
        except Exception as e:
            warnings.warn(f"pycbf reader failed: {e}; trying external tool fallback")

    # 3) external conversion
    print("reading via external conversion")
    return read_cbf_with_convert(path)


# ---------- embedding utilities ----------
def image_to_embedding(img: np.ndarray, method="pixel", size=224, device=None):
    """
    Convert numpy image (2D or 3D) to embedding vector.

    - method='pixel' : simple downsample -> flatten (fast, low-quality)
    - method='resnet' : use pretrained ResNet18 up to penultimate layer -> 512-d vector
    """
    if method == "pixel":
        # convert to PIL, resize, grayscale, flatten and normalize
        # print("### image", img, type(img), img.shape, img.size)
        pil = Image.fromarray(np.asarray(img).astype(np.uint8))
        pil = pil.convert("L").resize((size, size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        vec = arr.reshape(-1)
        # optionally normalize
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.astype(np.float32)

    if method == "resnet":
        if not HAVE_TORCH:
            raise RuntimeError("PyTorch not installed. Install torch and torchvision for resnet embeddings.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare model (ResNet18) and drop final classification layer
        model = models.resnet18(pretrained=True)
        # remove final fc to get features; get output of avgpool (512)
        model = nn.Sequential(*list(model.children())[:-1])  # outputs [B,512,1,1]
        model.to(device)
        model.eval()

        # transforms
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((size, size)),
            T.Grayscale(num_output_channels=3) if img.ndim == 2 else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # ensure img is HxW (gray) or HxWxC
        if img.ndim == 2:
            arr3 = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            arr3 = np.concatenate([img, img, img], axis=2)
        else:
            arr3 = img
        # convert to byte image if floats in [0,1] -> to 0..255
        if arr3.dtype == np.float32 or arr3.dtype == np.float64:
            amax = arr3.max() if arr3.max() > 0 else 1.0
            arr3 = (255.0 * (arr3 / amax)).astype(np.uint8)

        img_t = transforms(arr3)
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(img_t)  # shape [1,512,1,1]
            feat = feat.squeeze().cpu().numpy()
        # L2-normalize
        feat = feat.astype(np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-10)
        return feat

    raise ValueError("Unknown method: choose 'pixel' or 'resnet'.")


def cbf_to_vector(path, method="pixel", size=224):
    """
    High-level helper: read CBF at 'path' and return feature vector.
    method: 'pixel' or 'resnet'
    """
    img = read_cbf_image(path)
    vec = image_to_embedding(img, method=method, size=size)
    return vec

# ---------- Qdrant utilities ----------

def get_qdrant_client(host="localhost", port=6333, path=None):
    """
    If `path` is provided -> local embedded storage
    Else -> connect to host:port
    """
    if path:
        return QdrantClient(path=path)
    return QdrantClient(host=host, port=port)


def ensure_collection(client, collection_name, vector_size):
    """
    Create collection if not exists
    """
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
        )
    else:
        print(f"Collection '{collection_name}' already exists")


def ingest_cbf_to_qdrant(
    cbf_path: str,
    client: QdrantClient,
    collection_name: str = "cbf_images",
    method: str = "resnet",
    size: int = 224,
    extra_payload: dict | None = None
):
    """
    Convert CBF → vector → store in Qdrant
    """
    # print("### call ingest_cbf_to_qdrant", method)

    # read + embed
    img = read_cbf_image(cbf_path)
    vec = image_to_embedding(img, method=method, size=size)
    print("embedded vector", vec, len(vec))

    # ensure collection
    ensure_collection(client, collection_name, len(vec))

    # build payload
    payload = {
        "filename": os.path.basename(cbf_path),
        "path": os.path.abspath(cbf_path),
        "height": int(img.shape[0]),
        "width": int(img.shape[1]),
        "method": method,
    }
    if extra_payload:
        payload.update(extra_payload)

    # unique id
    point_id = str(uuid.uuid4())

    # upsert point
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload=payload
            )
        ],
    )

    print(f"✅ Ingested: {cbf_path} → {collection_name} [{point_id}]")
    return point_id


def search_similar_cbf(
    cbf_path: str,
    client: QdrantClient,
    collection_name: str = "cbf_images",
    method: str = "resnet",
    size: int = 224,
    limit: int = 5
):
    """
    Find similar images from Qdrant given a query CBF
    """
    img = read_cbf_image(cbf_path)
    vec = image_to_embedding(img, method=method, size=size)

    hits = client.search(
        collection_name=collection_name,
        query_vector=vec.tolist(),
        limit=limit
    )

    print("\n=== Similar images ===")
    for h in hits:
        print(f"Id: {h.id} Score: {h.score:.4f}  File: {h.payload.get('path')}")
    return hits

# PNG tools
###############################################################################
# IMAGE → PNG
###############################################################################

def image_to_png(
    img: np.ndarray,
    out_path: str,
    clip_percentile: float | None = None,
):
    """
    Save a 2D numpy image as a grayscale PNG.

    Parameters
    ----------
    img : np.ndarray
        2D array (H, W), int or float
    out_path : str
        Output PNG filename
    clip_percentile : float | None
        Optional percentile clipping (e.g. 99.5)
    """

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    print("orig", img)
    img = img.astype(np.float32)

    # Optional robust clipping (recommended for diffraction images)
    if clip_percentile is not None:
        lo = np.percentile(img, 100 - clip_percentile)
        hi = np.percentile(img, clip_percentile)
        img = np.clip(img, lo, hi)

    vmin = img.min()
    vmax = img.max()

    if vmax <= vmin:
        raise ValueError(f"Invalid intensity range: min={vmin}, max={vmax}")

    # Normalize to 0–255
    img_norm = (img - vmin) / (vmax - vmin)
    img_u8 = np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)
    print("img_u8", img_u8)

    pil = Image.fromarray(img_u8, mode="L")
    pil.save(out_path)

    return out_path

def cbf_to_png(
    cbf_path: str,
    out_png: str | None = None,
    clip_percentile: float | None = 99.5,
):
    """
    Convert a CBF file directly to a PNG image.

    Parameters
    ----------
    cbf_path : str
        Input CBF file
    out_png : str | None
        Output PNG path (defaults to cbf_path + '.png')
    clip_percentile : float | None
        Percentile clipping (recommended: 99–99.9)
    """

    if out_png is None:
        out_png = cbf_path + ".png"

    img = read_cbf_image(cbf_path)
    image_to_png(img, out_png, clip_percentile=clip_percentile)

    print(f"PNG written: {out_png}")
    return out_png


# Example CLI usage
def example_cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cbf_path")
    p.add_argument("--method", choices=["resnet", "pixel"], default="pixel")
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    vec = cbf_to_vector(args.cbf_path, method=args.method, size=args.size)
    print("vector length:", vec.shape)
    # Save to .npy for ingest into vector DB
    out = args.cbf_path + f".{args.method}.embed.npy"
    np.save(out, vec)
    print("Saved embedding to", out)


def cbf_to_qdrant():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to cbf file or directory of cbf files")
    p.add_argument("--method", default="resnet", choices=["resnet", "pixel"])
    p.add_argument("--vector-size", type=int, default=224, help="embedding vector size")
    p.add_argument("--collection", default="cbf_images")
    p.add_argument("--qdrant-path", default=None, help="Local Qdrant path (optional)")
    p.add_argument("--qdrant-host", default="localhost")
    p.add_argument("--qdrant-port", type=int, default=6333)
    p.add_argument("--search", action="store_true", help="Perform similarity search instead of ingest")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--write-png", action="store_true",
               help="Write PNG visualization of CBF")
    p.add_argument("--png-clip", type=float, default=99.5,
               help="Percentile clipping for PNG")


    args = p.parse_args()

    client = get_qdrant_client(
        path=args.qdrant_path,
        host=args.qdrant_host,
        port=args.qdrant_port
    )

    # single file or directory
    if os.path.isdir(args.path):
        files = glob(os.path.join(args.path, "*.cbf"))
    else:
        files = [args.path]

    if args.search:
        search_similar_cbf(
            files[0],
            client,
            collection_name=args.collection,
            method=args.method,
            size=args.size,
            limit=args.limit
        )
    elif args.write_png:
        cbf_to_png(args.path, clip_percentile=args.png_clip)
    else:
        for f in files:
            ingest_cbf_to_qdrant(
                f,
                client,
                collection_name=args.collection,
                method=args.method,
                size=args.vector_size
            )

# main
if __name__ == "__main__":
    cbf_to_qdrant()
