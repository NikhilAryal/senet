import crypten
import crypten.communicator as comm

crypten.init()
rank = comm.get().get_rank()
world_size = comm.get().get_world_size()