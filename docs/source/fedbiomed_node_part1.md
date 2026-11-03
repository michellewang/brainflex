# Fed-BioMed node (Part 1)

## Start the containers

```bash
docker compose up -d
```

This will download the images (the first time it is run) and start them:

```{code-block}
[+] Running 3/3
 ✔ Network brainflex-node_default   Created
 ✔ Container brainflex-vpn-default   Healthy
 ✔ Container brainflex-node-default  Started
```

The two images are:



2. Send the node's WireGuard public key to the VPN administrator, who registers it on the VPN server:

   ```bash
   docker compose logs vpn | grep 'public key'
   ```

   The key is kept in `vpn_config/wireguard/wg0.conf` and reused when the containers are restarted.

## Troubleshooting

Disk quota exceeded
