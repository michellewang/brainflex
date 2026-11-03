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
