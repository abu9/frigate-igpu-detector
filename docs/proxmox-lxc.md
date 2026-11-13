# Deploying in a Proxmox LXC

The goal: give the sidecar access to the iGPU's **render node** so Vulkan works, without
IOMMU/VFIO, without passing the whole GPU into a VM, and without giving the container more
privileges than necessary.

## What the container needs

| Resource | Needed? | Why |
| --- | --- | --- |
| `/dev/dri/renderD128` | **yes** | Vulkan compute (ncnn) |
| `/dev/dri/card0` / `card1` | no | only for display/KMS |
| `/dev/kfd` | no | ROCm compute only |
| IOMMU / VFIO | no | the host keeps the GPU; LXC shares it |
| X11 / Wayland | no | headless compute |
| `video` group | no | `render` is enough for compute |

The kernel driver (`amdgpu`) stays on the **host**. Mesa/RADV lives in the **container**.

## Host check

```bash
ls -l /dev/dri/
# crw-rw---- 1 root render 226, 128 ... renderD128

dmesg | grep -i amdgpu | tail -3
```

If `renderD128` is missing, fix the host driver first; nothing in this project can help.

## Recommended: Proxmox VE 8.2+ device passthrough

The `dev[n]` option creates the device node inside the container with the ownership you choose,
and sets up the device cgroup automatically. It works for unprivileged containers.

The examples use container ID `101`; replace it with your container's ID.

1. Find the GID of the container's `render` group (this is **not** necessarily the host GID):

   ```bash
   pct exec 101 -- getent group render
   # render:x:993:
   ```

2. Pass the device, using that GID:

   ```bash
   pct set 101 -dev0 /dev/dri/renderD128,gid=993,mode=0660
   pct reboot 101
   ```

   The same thing is available in the web UI under **Resources → Add → Device Passthrough**
   (tick *Advanced* to set GID and mode).

3. Verify inside the container:

   ```bash
   ls -l /dev/dri/
   # crw-rw---- 1 root render 226, 128 ... renderD128
   ```

   If you see `nobody nogroup`, the GID is wrong.

## Legacy: raw `lxc.*` configuration

This still works and is what most older guides use. On an **unprivileged** container the bind
mount keeps the host's ownership, and host GIDs are not mapped into the container, so the node
becomes `nobody:nogroup`. You need an `lxc.idmap` entry as well.

```ini
# /etc/pve/lxc/101.conf
lxc.cgroup2.devices.allow: c 226:128 rwm
lxc.mount.entry: /dev/dri/renderD128 dev/dri/renderD128 none bind,optional,create=file
```

For an unprivileged container, map the host `render` GID (say 993 on the host) to the same GID
inside the container:

```ini
lxc.idmap: u 0 100000 65536
lxc.idmap: g 0 100000 993
lxc.idmap: g 993 993 1
lxc.idmap: g 994 100994 64542
```

Rules:

- The `g` ranges must cover `0..65535` with no gaps or overlaps.
- The host GID you map (993 here) must be delegated in `/etc/subgid`, e.g. `root:993:1`.
- A custom `lxc.idmap` replaces Proxmox's default mapping for that container; back up
  `/etc/pve/lxc/101.conf` and `/etc/subgid` first.
- On a cluster, `/etc/subgid` is per node.

If this feels fragile, use the `dev0` method above — it avoids all of it.

## Container sizing

| Resource | Recommendation | Observed |
| --- | --- | --- |
| CPU | 2 vCPU | sidecar uses ~0.05 core on the iGPU (startup/conversion is the peak) |
| RAM | 1 GB | 120–220 MB |
| Disk | 2–4 GB | venv ~310 MB + models ~120 MB + pnnx ~85 MB + OS |
| Privileged container | **no** | unprivileged works with `dev0` passthrough |

## Container packages

```bash
apt update
apt install -y python3 python3-venv git ca-certificates curl \
  mesa-vulkan-drivers libvulkan1 vulkan-tools
```

Notes:

- `python3-venv` provides `ensurepip`; a system `python3-pip` is optional because the venv has
  its own pip.
- `unzip` is not needed: the pnnx release is extracted with `python3 -m zipfile`.
- `libgomp1` is bundled inside the ncnn wheel, so it does not need to be installed explicitly.
- Do **not** install ROCm, CUDA, Xorg, or `libdrm-dev`; none are used.

Verify the driver:

```bash
vulkaninfo --summary | sed -n '/Devices:/,/^$/p'
```

You want a real device:

```text
deviceName = AMD Radeon Graphics (RADV RENOIR)
driverName = radv
```

`llvmpipe (LLVM ...)` is the CPU software rasteriser. If that is the only device, the sidecar
will refuse to start on the Vulkan backend (a warmup > 5 s is treated as llvmpipe) and fall back
to OpenVINO.

## Service user

The sidecar runs as an unprivileged system user. It needs `render` membership:

```bash
useradd --system --home /var/lib/frigate-detector --shell /usr/sbin/nologin frigate-detector
usermod -aG render frigate-detector
```

In systemd:

```ini
SupplementaryGroups=render
```

Alternatively, make the container's `render` GID match the device node's GID (the `dev0`
method with `gid=` does this for you) and skip the group membership.

## Sharing the iGPU

Multiple containers can share `renderD128`; the GPU is time-sliced. Keep an eye on:

```bash
cat /sys/class/drm/card1/device/gpu_busy_percent
```

A 320×320 YOLOv9-m model at ~15 inferences/s uses roughly 70–80 % of a Vega 7. Add Jellyfin
transcoding or a desktop session and latency will rise.

## Security notes

- Keep the container **unprivileged**.
- Passing a render node gives the container the DRM ioctl interface. That is a real kernel
  attack surface; treat the container as trusted.
- Do not set `lxc.apparmor.profile: unconfined` unless you have no alternative; prefer fixing
  the GID mapping. If AppArmor blocks the device, check `dmesg` and the container's profile.
- The ZMQ port has no authentication. Firewall it to your Frigate host.

## Frigate in a VM?

That is fine and is the point of the split architecture: Frigate (Docker or VM) talks to
`tcp://<lxc-ip>:5555` and never touches a GPU. Only the LXC running this sidecar needs
`/dev/dri/renderD128`.
