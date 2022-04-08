# Basars gRPC

gRPC server/client implementation for Basars models

## Installing dependencies
```bash
python -m pip install -r requirements.txt
```

## Running server
```bash
python -m basars_grpc_server.server
```

#### Environmental Variables
| Key                   | Default Value |
|-----------------------|---------------|
| `BASARS_GRPC_HOST`    | [::]          |
| `BASARS_GRPC_PORT`    | 9000          |
| `BASARS_POOL_WORKERS` | 10            |

## Running client
```bash
python -m basars_grpc_client.client
```

### Environmental Variables
| Key                       | Default Value |
|---------------------------|---------------|
| `BASARS_HOST`             | localhost     |
| `BASARS_PORT`             | 9000          |
| `BASARS_IMAGE_SOURCE_DIR` | sample_images |
| `BASARS_IMAGE_TARGET_DIR` | target_images |
