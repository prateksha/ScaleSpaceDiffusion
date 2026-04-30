"""
Helpers for distributed training using torchrun / torch.distributed.
"""

import io
import os
import random
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

def setup_dist():
    """
    Setup a torch.distributed process group.
    Works with torchrun launcher.
    """
    if dist.is_initialized():
        return

    # Get rank, local rank, world size from environment (torchrun sets these)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set device for this process
    device = th.device(f"cuda:{local_rank}" if th.cuda.is_available() else "cpu")
    # Set backend
    backend = "nccl" if th.cuda.is_available() else "gloo"
    # os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    # if "MASTER_PORT" not in os.environ:
    #     os.environ["MASTER_PORT"] = str(random.randint(12000, 20000))

    # Initialize the process group
    th.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # if th.cuda.is_available():
    #     th.cuda.set_device(local_rank)
    #     dist.barrier(device_ids=[local_rank])
    # else:
    #     dist.barrier()


def dev():
    """
    Get the device to use for this process.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if th.cuda.is_available():
        return th.device(f"cuda:{local_rank}")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    Uses torch.distributed.broadcast.
    """
    chunk_size = 2 ** 30  # 1GB chunks
    rank = dist.get_rank() if dist.is_initialized() else 0
    # print(f"load_state_dict: {rank}, {dev()}")
    
    if rank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
    else:
        data = b""
        num_chunks = 0

    if dist.is_initialized():
        num_chunks_tensor = th.tensor([num_chunks], dtype=th.long, device=dev())
        # print(f"Num chunks tensor: ", num_chunks_tensor.shape)
        dist.broadcast(num_chunks_tensor, src=0)
        num_chunks = num_chunks_tensor.item()
        # print(f"Total chunks: {num_chunks}")

        for i in range(num_chunks):
            # print(f"Processing chunk {i}/{num_chunks}")
            
            if rank == 0:
                actual_chunk_size = min(chunk_size, len(data) - i * chunk_size)
                chunk_size_tensor = th.tensor([actual_chunk_size], dtype=th.long, device=dev())
            else:
                chunk_size_tensor = th.tensor([0], dtype=th.long, device=dev())
            
            dist.broadcast(chunk_size_tensor, src=0)
            actual_chunk_size = chunk_size_tensor.item()
            # print(f"Chunk {i} size: {actual_chunk_size}")
            
            if rank == 0:
                chunk_data = data[i * chunk_size:i * chunk_size + actual_chunk_size]
                chunk = th.tensor(list(chunk_data), dtype=th.uint8, device=dev())
            else:
                chunk = th.empty(actual_chunk_size, dtype=th.uint8, device=dev())
            
            dist.broadcast(chunk, src=0)
            
            if rank != 0:
                data += bytes(chunk.cpu().tolist())  # Move to CPU before converting to bytes
            
            # print(f"Completed chunk {i}")
    # print(f"Loading state dict, data size: {len(data)}")
    # print(f"Returning from load_state_dict: {rank}, {dev()}, {kwargs['map_location']}")
    return th.load(io.BytesIO(data), **kwargs)

# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across ranks.
#     Uses torch.distributed.broadcast.
#     """
#     chunk_size = 2 ** 30  # 1GB chunks
#     rank = dist.get_rank() if dist.is_initialized() else 0
#     print(f"load_state_dict: {rank}, {dev()}")
#     if rank == 0:
#         with bf.BlobFile(path, "rb") as f:
#             data = f.read()
#         num_chunks = len(data) // chunk_size
#         if len(data) % chunk_size:
#             num_chunks += 1
#     else:
#         data = b""
#         num_chunks = 0

#     # Broadcast number of chunks
#     print("Broadcasting")
#     if dist.is_initialized():
#         num_chunks_tensor = th.tensor([num_chunks], dtype=th.long, device=dev())
#         dist.broadcast(num_chunks_tensor, src=0)
#         num_chunks = num_chunks_tensor.item()

#         for i in range(num_chunks):
#             print(f"chunk {i}/num_chunks")
#             if rank == 0:
#                 chunk = th.tensor(list(data[i * chunk_size:(i + 1) * chunk_size]), dtype=th.uint8, device=dev())
#             else:
#                 chunk = th.empty(min(chunk_size, len(data) - i * chunk_size if rank == 0 else chunk_size), dtype=th.uint8, device=dev())
#             dist.broadcast(chunk, src=0)
#             if rank != 0:
#                 data += bytes(chunk.tolist())

#     return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if dist.is_initialized():
        for p in params:
            with th.no_grad():
                dist.broadcast(p, src=0)
