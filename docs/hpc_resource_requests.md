# Economical HPC resource requests

Resource sizing is part of every job plan and submission review. Use the
smallest practical allocation that can reliably finish the selected work with
a justified safety margin. This applies to batch scripts, interactive
allocations, and commands proposed in conversation.

## Before submission

1. Check existing artifacts and compatible resume caches. Estimate remaining
   work after filtering, deduplication, and reuse, rather than sizing a run from
   the total corpus or the number of source pieces alone.
2. Validate inputs before allocating a GPU. Run metadata audits, XML validation,
   feature extraction, and other CPU-only stages on CPU compute resources.
   Keep substantial processing off management/login nodes.
3. Check how the implementation uses resources. A sequential pipeline does not
   automatically benefit from many CPUs, and extra GPUs only help when the code
   explicitly supports them. Prefer one GPU for inference unless measured
   scaling justifies more.
4. Use a comparable completed job or a small representative pilot to estimate
   memory and elapsed time. Account for model startup, preprocessing, input
   length, and final artifact writes. Retain pilot outputs when compatible with
   the full run.
5. Choose a realistic walltime and sufficient memory headroom. Document the
   workload, resource request, supporting measurements, and remaining
   uncertainty in the run notes or submission instructions. If measurements are
   unavailable, label the initial request as an estimate and refine it after
   the pilot.

Do not treat old `#SBATCH` values as universal defaults. Host RAM and GPU memory
are different resources; lowering a host RAM request does not reduce the
model's GPU memory requirement. Accounting measurements may not capture every
brief peak or represent the aggregate usage of a multi-task job.

## Preserve progress and avoid wasted allocation

Use existing resumable pipelines with bounded chunks and atomic progress
writes. A shorter allocation is useful only if interrupted work can be resumed
without repeating most of the computation. Excessively small limits that
repeatedly prevent model startup or completion are not economical.

Where practical, keep CPU preparation and feature extraction outside the GPU
allocation. Avoid rerunning compatible prompt or score embeddings solely
because another stage failed. Check provenance before reusing cached results.

## Investigate queues before changing requests

Inspect the pending reason, requested resources, and actual GPU allocations.
For example, these are read-only diagnostics:

```bash
scontrol show job JOBID
sprio -j JOBID -l
squeue --start -j JOBID
sinfo -p gpu -N -O NodeList:15,Gres:35,GresUsed:50
```

Replace `JOBID` with the numeric job ID. GPU capacity is not the same as GPU
availability: a node can have idle CPUs while all its GPUs are allocated.
Start-time estimates can change and are not guarantees.

Reduce requests when evidence supports the change. Preserve an existing queued
job through a supported pending-job update where possible, rather than
cancelling and resubmitting without a reason. Do not keep reducing memory or
walltime when the actual bottleneck is an occupied GPU.

## Learn from completed runs

Inspect elapsed time and memory accounting, including batch/step rows:

```bash
sacct -j JOBID \
  --format=JobID,JobName%25,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,AllocTRES
```

Use successful representative runs to refine subsequent requests. An early
failure before model loading is not evidence of the full workload's memory or
runtime needs. Treat measurements from different models or substantially
different input lengths with caution.

Continue to follow the dataset preflight and artifact inventory rules in
[AGENTS.md](../AGENTS.md). Any workflow that may change embedding outputs must
refresh the inventory as its final step, including failure paths, and commit
the lightweight inventory files. Requesting fewer resources does not waive
those requirements.
