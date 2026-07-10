from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import PrivateL1PrivateL2CacheHierarchy
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.resources.resource import CustomResource
from gem5.simulate.simulator import Simulator

import sys
l1_size = sys.argv[3] if len(sys.argv) > 3 else "64KiB"
cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size=l1_size,
    l1i_size="64KiB",
    l2_size="256KiB"
)

memory = SingleChannelDDR4_2400()

import sys

cpu_type_arg = sys.argv[2] if len(sys.argv) > 2 else "minor"
cpu_type = CPUTypes.MINOR if cpu_type_arg == "minor" else CPUTypes.TIMING

# CPU Pipeline
processor = SimpleProcessor(cpu_type=cpu_type, isa=ISA.RISCV, num_cores=1)

# Binary compiled with -march=rv32imafc: RV32, no vector extension
for core in processor.cores:
    for isa_obj in core.core.isa:
        isa_obj.riscv_type = "RV32"
        isa_obj.enable_rvv = False

def make_hdc_fu_pool():
    from m5.objects import (MinorFU, MinorFUPool, MinorOpClassSet, MinorOpClass,
                            MinorDefaultIntFU, MinorDefaultIntMulFU,
                            MinorDefaultIntDivFU, MinorDefaultFloatSimdFU,
                            MinorDefaultPredFU, MinorDefaultMemFU,
                            MinorDefaultMiscFU)
    hdc_fu = MinorFU()
    hdc_fu.opClasses = MinorOpClassSet(opClasses=[MinorOpClass(opClass="HdcAlu")])
    hdc_fu.opLat = 2
    hdc_fu.issueLat = 1
    
    pool = MinorFUPool()
    pool.funcUnits = [
        MinorDefaultIntFU(),
        MinorDefaultIntFU(),
        MinorDefaultIntMulFU(),
        MinorDefaultIntDivFU(),
        MinorDefaultFloatSimdFU(),
        MinorDefaultPredFU(),
        MinorDefaultMemFU(),
        MinorDefaultMiscFU(),
        hdc_fu,
    ]
    return pool

if cpu_type == CPUTypes.MINOR:
    for core in processor.cores:
        core.core.executeFuncUnits = make_hdc_fu_pool()

board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

import sys
binary_path = sys.argv[1] if len(sys.argv) > 1 else "software/build/kernel_HDC.elf"
binary = CustomResource(binary_path)

board.set_se_binary_workload(binary)

simulator = Simulator(board=board)

print("Starting HDC-RFF RISC-V simulation (TimingCPU)...")
simulator.run()
print(f"Simulation finished at tick {simulator.get_current_tick()}")
