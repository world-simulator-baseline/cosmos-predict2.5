from unittest import mock

import torch

from cosmos_predict2._src.imaginaire.trainer import ImaginaireTrainer


def _trainer_without_initialization() -> ImaginaireTrainer:
    trainer = ImaginaireTrainer.__new__(ImaginaireTrainer)
    trainer._graceful_stop_requested = False
    trainer._graceful_stop_signal = None
    trainer._graceful_stop_announced = False
    return trainer


def test_graceful_stop_signal_handler_only_records_request() -> None:
    trainer = _trainer_without_initialization()

    trainer._request_graceful_checkpoint_stop(10, None)

    assert trainer._graceful_stop_requested
    assert trainer._graceful_stop_signal == 10


def test_graceful_stop_sync_propagates_request_from_another_rank() -> None:
    trainer = _trainer_without_initialization()

    def emulate_remote_request(value: torch.Tensor, op) -> None:
        assert op == torch.distributed.ReduceOp.MAX
        value.fill_(1)

    with (
        mock.patch("cosmos_predict2._src.imaginaire.trainer.dist.is_available", return_value=True),
        mock.patch("cosmos_predict2._src.imaginaire.trainer.dist.is_initialized", return_value=True),
        mock.patch("cosmos_predict2._src.imaginaire.trainer.dist.get_backend", return_value="gloo"),
        mock.patch(
            "cosmos_predict2._src.imaginaire.trainer.dist.all_reduce",
            side_effect=emulate_remote_request,
        ) as all_reduce,
        mock.patch("cosmos_predict2._src.imaginaire.trainer.log.warning") as warning,
    ):
        assert trainer._sync_graceful_checkpoint_stop()
        assert trainer._sync_graceful_checkpoint_stop()

    assert trainer._graceful_stop_requested
    assert all_reduce.call_count == 2
    warning.assert_called_once()
