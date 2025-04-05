from .inlineif import _DTC_inline_if

class DTCTraspiler(_DTC_inline_if):
    def __init__(self, model, model_name, method = "inline"):
        match method:
            case 'inline':
                super(_DTC_inline_if).__init__(model, model_name)
            case default:
                raise Exception(f"{method} not found")