import torch


class Context:
    """Cross-round state for federated training with one shared global model."""

    def __init__(
        self,
        users_num: int,
        model: torch.nn.Module,
        class_names: list[str],
        results_dir: str = "",
        learning_rate: float = 0.01,
    ):
        self.model = model
        self.results_dir = results_dir
        self.users_num = users_num
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.learning_rate = float(learning_rate)
        self.users = []
        self.base_model_state: dict[int, dict[str, torch.Tensor]] = {}
        self.updated_model_state: dict[int, dict[str, torch.Tensor]] = {}
        self.client_gradients: dict[int, dict[str, torch.Tensor]] = {}
        self.new_model_state: dict[int, dict[str, torch.Tensor]] = {}
        self.protocol_messages: dict[int, dict] = {}
        self.user_selected: list[int] = []
        self.samples_num: list[int] = []
        self.update_sample_counts: dict[int, int] = {}
        self.aggregation_weights: dict[int, float] = {}
        self.trainable_param_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.glob_iter = 0

    def get_base_model_state(self) -> dict[str, torch.Tensor]:
        return self.base_model_state[0]

    def set_base_model_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base_model_state[0] = state_dict

    def set_updated_model_state(
        self, user_id: int, state_dict: dict[str, torch.Tensor]
    ) -> None:
        self.updated_model_state[user_id] = state_dict

    def continue_to_next_round(self) -> None:
        self.base_model_state = self.new_model_state
        self.new_model_state = {}
        self.updated_model_state = {}
        self.client_gradients = {}
        self.user_selected = []
        self.protocol_messages = {}
        self.update_sample_counts = {}
        self.aggregation_weights = {}
