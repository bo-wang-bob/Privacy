import torch


class Context:
    """Cross-round state for global or personalized federated prompt tuning."""

    def __init__(
        self,
        users_num: int,
        model: torch.nn.Module,
        class_names: list[str],
        results_dir: str = "",
        mode: str = "centralized",
        learning_rate: float = 0.01,
    ):
        if mode not in {"centralized", "local"}:
            raise ValueError("mode must be 'centralized' or 'local'.")
        self.model = model
        self.results_dir = results_dir
        self.users_num = users_num
        self.mode = mode
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.learning_rate = float(learning_rate)
        self.users = []
        self.base_model_state: dict[int, dict[str, torch.Tensor]] = {}
        self.updated_model_state: dict[int, dict[str, torch.Tensor]] = {}
        self.new_model_state: dict[int, dict[str, torch.Tensor]] = {}
        self.user_selected: list[int] = []
        self.samples_num: list[int] = []
        self.trainable_param_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.glob_iter = 0

    def get_base_model_state(self, user_id: int) -> dict[str, torch.Tensor]:
        return self.base_model_state[0 if self.mode == "centralized" else user_id]

    def get_new_model_state(self, user_id: int) -> dict[str, torch.Tensor]:
        return self.new_model_state[0 if self.mode == "centralized" else user_id]

    def set_base_model_state(
        self, user_id: int, state_dict: dict[str, torch.Tensor]
    ) -> None:
        if self.mode == "centralized":
            if user_id == 0:
                self.base_model_state[0] = state_dict
        else:
            self.base_model_state[user_id] = state_dict

    def set_updated_model_state(
        self, user_id: int, state_dict: dict[str, torch.Tensor]
    ) -> None:
        self.updated_model_state[user_id] = state_dict

    def continue_to_next_round(self) -> None:
        self.base_model_state = self.new_model_state
        self.new_model_state = {}
        self.updated_model_state = {}
        self.user_selected = []
