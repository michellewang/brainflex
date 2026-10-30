# Fed-BioMed node

<!-- TODO split into Part 1 and Part 2 -->

All commands on this page assume the working directory is the root of the `brainflex-node` repository.

The node runs with Docker Compose as two containers:

- `vpn`: a WireGuard client that connects to the brainflex VPN server
- `node`: the published [`fedbiomed/node`](https://hub.docker.com/r/fedbiomed/node) image, using the network of the `vpn` container to reach the researcher

## Requirements

- Docker with the Compose plugin (`docker compose`)
- Outbound UDP access to the VPN server
- The Nipoppy study downloaded with `brainflex get-data` (see [Initial setup](initial_setup.md)), which the node mounts

## Setup

<!-- TODO config that config.env is where it is expected -->

1. Build and start the containers:

   ```bash
   docker compose up -d --build
   ```

2. Send the node's WireGuard public key to the VPN administrator, who registers it on the VPN server:

   ```bash
   docker compose logs vpn | grep 'public key'
   ```

   The key is kept in `vpn_config/wireguard/wg0.conf` and reused when the containers are restarted.

3. Once the key is registered, check that the VPN server and the researcher are reachable, and that the node is running:

   ```bash
   scripts/vpn_status
   docker compose exec -u fedbiomed node fedbiomed node -p /fbm-node status
   ```

## Usage

Fed-BioMed node files (configuration, database, data) are in `fbm-node`, mounted at `/fbm-node` in the `node` container.

Register the datasets [created by `nipoppy2fedbiomed`](fedbiomed_dataset.md):

```bash
docker compose exec -u fedbiomed node fedbiomed node -p /fbm-node dataset add --file /nipoppy_study/derivatives/nipoppy2fedbiomed/0.1.0/output/age/dataset_info.json
docker compose exec -u fedbiomed node fedbiomed node -p /fbm-node dataset add --file /nipoppy_study/derivatives/nipoppy2fedbiomed/0.1.0/output/sex/dataset_info.json
```

<!-- TODO -->
```bash
docker compose exec -u fedbiomed node fedbiomed node -p /fbm-node training-plan view
docker compose exec -u fedbiomed node fedbiomed node -p /fbm-node training-plan approve
```

Check that they are listed:

```bash
docker compose exec -u fedbiomed node fedbiomed node -p /fbm-node dataset list
```

Stop the containers:

```bash
docker compose down
```
