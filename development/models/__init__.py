def create_model(opt, rank):
    from .mesh_classifier import DistributedClassifierModel
    model = DistributedClassifierModel(opt, rank)
    return model