# Security policy

## Supported versions

Security fixes are provided for the latest release (currently `0.1.x`).

## Threat model

This project is a network service that runs a model on a GPU. It is intended for
a trusted home network, not for exposure to the internet.

Things to know before deploying:

- **The ZMQ endpoint has no authentication.** Anyone who can reach the port can send
  inference requests and model transfers. Bind it to a LAN address and firewall it to
  the Frigate host.
- **Model uploads are trusted input.** Frigate sends ONNX bytes over the same socket.
  The server writes them to disk and runs `pnnx` on them. A malicious peer could upload a
  hostile model. Keep the port on a trusted network and keep `FRIGATE_ZMQ_MAX_MODEL_MB`
  as small as practical.
- **The render node is a kernel attack surface.** `/dev/dri/renderD128` exposes the DRM
  ioctl interface. A container with the render node is still sharing the host kernel.
  Prefer unprivileged containers and avoid `lxc.apparmor.profile: unconfined`.
- **The service runs as a dedicated unprivileged user.** Do not run it as root. Grant
  only the `render` group.
- **Model weights are not distributed with this project.** Users supply their own ONNX
  files and are responsible for their licences.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting (Security → Report a vulnerability)
instead of a public issue. Include:

- affected version (`fzd --version`),
- backend and GPU/driver,
- a minimal reproduction,
- relevant log lines.
