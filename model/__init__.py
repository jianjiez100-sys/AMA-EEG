from .models import Conv_att_simple_new
from .pl_models import ExtractorModel


def get_extractor_class(dataset_name):
    """Select the original experiment rather than sharing its training logic."""
    if dataset_name == 'SEED':
        from .seed_pl_models import ExtractorModel as SEEDExtractorModel
        return SEEDExtractorModel
    if dataset_name == 'FACED':
        return ExtractorModel
    raise ValueError(f'No source pretraining protocol for {dataset_name}')
