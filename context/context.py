import torch

from users.user import UserBase


class Context:
    """Cross-round state shared by the FPL clients and seismograph aggregator."""

    def __init__(
        self,
        users_num: int,
        model: torch.nn.Module,
        class_names: list[str],
        results_dir: str = "",
        mode: str = "centralized",
        learning_rate: float = 0.01,
        fpl: bool = True,
    ):
        if not fpl:
            raise ValueError("This branch only supports federated prompt learning.")
        if mode not in {"centralized", "local"}:
            raise ValueError(
                "FPL seismograph supports only 'centralized' and 'local' training modes; "
                f"got {mode!r}."
            )

        self.model = model
        self.results_dir = results_dir
        self.users_num = users_num
        self.mode = mode
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.learning_rate = float(learning_rate)
        self.users: list[UserBase] = []

        self.base_model_state: dict[int, dict[str, torch.Tensor]] = {}
        self.updated_model_state: dict[int, dict[str, torch.Tensor]] = {}
        self.new_model_state: dict[int, dict[str, torch.Tensor]] = {}
        self.user_selected: list[int] = []
        self.samples_num: list[int] = []
        self.fpl = True
        self.text_feature_dict: dict[int, torch.Tensor] = {}
        self.poison_label = None

        self.trainable_param_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        self.running_statistics_names: list[str] = []
        self.glob_iter = 0
        self.attack_rounds: list[int] = []

    def get_base_model_state(self, user_id: int) -> dict[str, torch.Tensor]:
        if self.mode == "centralized":
            return self.base_model_state[0]
        return self.base_model_state[user_id]

    def get_new_model_state(self, user_id: int) -> dict[str, torch.Tensor]:
        if self.mode == "centralized":
            return self.new_model_state[0]
        return self.new_model_state[user_id]

    def get_updated_model_state(self, user_id: int) -> dict[str, torch.Tensor]:
        return self.updated_model_state[user_id]

    def set_base_model_state(
        self,
        user_id: int,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        if self.mode == "centralized":
            if user_id == 0:
                self.base_model_state[0] = state_dict
        else:
            self.base_model_state[user_id] = state_dict

    def set_updated_model_state(
        self,
        user_id: int,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.updated_model_state[user_id] = state_dict

    def continue_to_next_round(self) -> None:
        self.base_model_state = self.new_model_state
        self.new_model_state = {}
        self.updated_model_state = {}
        self.user_selected = []
        self.text_feature_dict = {}
