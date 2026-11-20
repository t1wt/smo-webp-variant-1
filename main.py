import argparse
import sys

from smo import Config, ExperimentRunner, Simulation


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def print_table(headers, rows):
    cells = [list(headers), *([str(cell) for cell in row] for row in rows)]
    widths = [max(map(len, column)) for column in zip(*cells)]
    for i, row in enumerate(cells):
        print(" | ".join(cell.rjust(width) for cell, width in zip(row, widths)))
        if i == 0:
            print("-+-".join("-" * width for width in widths))


def print_state(model: Simulation):
    print(f"\nt = {number(model.now)} с")
    for action in model.stats.actions:
        print(f"  {action}")

    for source, counts in zip(model.sources, model.stats.sources):
        print(f"И{source.id}  λ={number(source.rate)}  создано={counts.arrived}  "
              f"отказов={counts.rejected}  обслужено={counts.completed}  "
              f"последний результат: {source.last_result}")

    slots = "  ".join(
        f"{index}: {request.label if request else 'пусто'}"
        for index, request in enumerate(model.buffer.slots)
    )
    print(f"Буфер [{slots}]  указатель → {model.buffer.pointer}")

    for device in model.pool.devices:
        status = (f"{device.current.label} до t={number(device.finish_at)}"
                  if device.current else "свободен")
        print(f"IMG-{device.id}: {status}")
    print(f"Доля отказов: {model.refusal_rate:.2%}" if model.generated else "Доля отказов: —")


def print_results(model: Simulation):
    print("Характеристики источников")
    print_table(
        ("Источник", "Заявок", "pотк", "Tпреб, с", "TБП, с", "Tобсл, с", "ДБП, с²", "Добсл, с²"),
        (
            (f"И{i}", stats.arrived, number(stats.refusal_rate),
             *(number(value) for value in stats.time_metrics()))
            for i, stats in enumerate(model.stats.sources, 1)
        ),
    )
    print("\nХарактеристики приборов")
    print_table(
        ("Прибор", "Коэффициент использования"),
        ((f"IMG-{i}", number(busy / model.now))
         for i, busy in enumerate(model.stats.busy_time, 1)),
    )


def run_stepwise(config: Config, request_limit: int):
    model = Simulation(config, request_limit)
    print_state(model)
    while not model.finished:
        command = input("\nEnter — следующий шаг, q — выход: ").strip().lower()
        if command == "q":
            return
        if command:
            continue
        model.step()
        print_state(model)
    print("\nПрогон завершён.")


def main() -> int:
    parser = argparse.ArgumentParser(description="СМО сервиса WebP-миниатюр")
    modes = parser.add_subparsers(dest="mode", required=True, title="режимы")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--rates", type=float, nargs="+", default=[1.0, 1.0, 1.0],
                        metavar="λ", help="интенсивности источников в заявках/с")
    common.add_argument("--devices", type=int, default=2, help="число приборов")
    common.add_argument("--buffer", type=int, default=3, help="число ячеек буфера")
    common.add_argument("--tmin", type=float, default=1.0, help="минимальное время обслуживания, с")
    common.add_argument("--tmax", type=float, default=2.0, help="максимальное время обслуживания, с")
    common.add_argument("--seed", type=int, default=42, help="начальное число генераторов")

    step = modes.add_parser("step", parents=[common], help="пошаговый режим")
    step.add_argument("--requests", type=int, default=10, help="число генерируемых заявок")
    auto = modes.add_parser("auto", parents=[common], help="автоматический режим")
    auto.add_argument("--max-requests", type=int, default=100_000,
                      help="верхний предел числа заявок в одном прогоне")

    args = parser.parse_args()
    try:
        config = Config(tuple(args.rates), args.devices, args.buffer, args.tmin, args.tmax, args.seed)
        if args.mode == "step":
            run_stepwise(config, args.requests)
        else:
            result = ExperimentRunner().run_automatic(config, args.max_requests)
            print_results(result.model)
            if not result.converged:
                print(
                    f"На {result.model.generated} заявках заданная точность оценки "
                    "отказов не подтверждена.",
                    file=sys.stderr,
                )
                return 1
    except ValueError as error:
        parser.error(str(error))
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
    # python3 main.py step --rates 1 1 1 --devices 2 --buffer 3 --tmin 1 --tmax 2 --seed 42 --requests 10
    # python3 main.py auto --rates 1 1 1 --devices 2 --buffer 3 --tmin 1 --tmax 2 --seed 42 --max-requests 100000
