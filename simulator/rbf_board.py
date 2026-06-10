from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import PrivateL1PrivateL2CacheHierarchy
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.resources.resource import CustomResource 
from gem5.simulate.simulator import Simulator

cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="64KiB",  # L1 Data Cache
    l1i_size="64KiB",  # L1 Instruction Cache
    l2_size="256KiB"   # L2 Shared Cache
)

memory = SingleChannelDDR4_2400()

processor = SimpleProcessor(cpu_type=CPUTypes.TIMING, isa=ISA.RISCV, num_cores=1)

board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)



binary = CustomResource("software/build/rbf_kernel.elf")

board.set_se_binary_workload(binary)

simulator = Simulator(board=board)

print("Starting custom RISC-V hardware simulation...")
simulator.run()
print(f"Simulation finished perfectly at tick {simulator.get_current_tick()}")
