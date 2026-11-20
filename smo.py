from dataclasses import dataclass, field
from enum import IntEnum
from heapq import heapify, heappop, heappush
from math import ceil, isfinite
from random import Random


@dataclass(frozen=True)
class Config:
    source_rates: tuple[float, ...] = (1.0, 1.0, 1.0)
    device_count: int = 2
    buffer_capacity: int = 3
    tmin: float = 1.0
    tmax: float = 2.0
    seed: int = 42

    def __post_init__(self):
        if not self.source_rates or any(not isfinite(rate) or rate <= 0
                                        for rate in self.source_rates):
            raise ValueError("Нужна положительная интенсивность.")
        if self.device_count < 1 or self.buffer_capacity < 1:
            raise ValueError("Число приборов и вместимость буфера должны быть не меньше 1.")
        if not (0 < self.tmin <= self.tmax and isfinite(self.tmax)):
            raise ValueError("Границы времени должны удовлетворять 0 < tmin ≤ tmax < ∞.")


@dataclass(frozen=True)
class Request:
    id: int
    source_id: int
    image_uri: str
    arrival_at: float

    @property
    def label(self) -> str:
        return f"q{self.id}"


@dataclass
class Source:
    id: int
    rate: float
    rng: Random
    last_result: str = "—"

    def generate(self, request_id: int, now: float) -> Request:
        return Request(request_id, self.id, f"images/{request_id}.jpg", now)

    def next_arrival(self, now: float) -> float:
        return now + self.rng.expovariate(self.rate)

    def notify_rejected(self, request: Request):
        self.last_result = f"{request.label}: 503 OVERLOAD"

    def notify_completed(self, request: Request, result_uri: str):
        self.last_result = f"{request.label}: {result_uri}"


class EventKind(IntEnum):
    COMPLETION = 0
    ARRIVAL = 1


@dataclass(order=True, frozen=True)
class Event:
    time: float
    kind: EventKind
    target_id: int


class EventCalendar:
    def __init__(self):
        self.events: list[Event] = []

    def schedule(self, kind: EventKind, time: float, target_id: int):
        if not isfinite(time):
            raise ValueError("Модельное время вышло за диапазон чисел. Проверьте параметры.")
        heappush(self.events, Event(time, kind, target_id))

    def pop_next(self) -> Event:
        return heappop(self.events)

    def stop_arrivals(self):
        self.events = [event for event in self.events if event.kind == EventKind.COMPLETION]
        heapify(self.events)


@dataclass(frozen=True)
class OfferResult:
    slot: int
    evicted: Request | None


class RingBuffer:
    def __init__(self, capacity: int):
        self.slots: list[Request | None] = [None] * capacity
        self.pointer = 0
        self.count = 0

    def offer(self, request: Request) -> OfferResult:
        capacity = len(self.slots)
        slot = self.pointer
        if self.count < capacity:
            while self.slots[slot] is not None:
                slot = (slot + 1) % capacity
            self.count += 1
        evicted = self.slots[slot]
        self.slots[slot] = request
        self.pointer = (slot + 1) % capacity
        return OfferResult(slot, evicted)

    def poll_oldest(self) -> Request:
        occupied = ((index, req) for index, req in enumerate(self.slots) if req is not None)
        slot, request = min(occupied, key=lambda pair: (pair[1].arrival_at, pair[1].id))
        self.slots[slot] = None
        self.count -= 1
        return request


@dataclass
class Device:
    id: int
    tmin: float
    tmax: float
    rng: Random
    current: Request | None = None
    finish_at: float | None = None

    def start(self, request: Request, now: float) -> float:
        self.current = request
        self.finish_at = now + self.rng.uniform(self.tmin, self.tmax)
        return self.finish_at

    def finish(self) -> tuple[Request, str]:
        request = self.current
        assert request is not None
        self.current = self.finish_at = None
        return request, f"thumbnails/{request.id}.webp"


class DevicePool:
    def __init__(self, devices: list[Device]):
        self.devices = devices

    def first_free(self) -> Device | None:
        return next((device for device in self.devices if device.current is None), None)

    def start(self, device: Device, request: Request, now: float) -> float:
        return device.start(request, now)

    def finish(self, device_id: int) -> tuple[Request, str]:
        return self.devices[device_id - 1].finish()


@dataclass
class SourceStats:
    arrived: int = 0
    rejected: int = 0
    completed: int = 0
    wait_sum: float = 0.0
    wait_squares: float = 0.0
    service_sum: float = 0.0
    service_squares: float = 0.0
    rejected_wait_sum: float = 0.0

    @property
    def refusal_rate(self) -> float | None:
        return self.rejected / self.arrived if self.arrived else None

    def time_metrics(self) -> tuple[float | None, ...]:
        if not self.completed:
            return (None,) * 5
        wait = self.wait_sum / self.completed
        service = self.service_sum / self.completed
        return (
            wait + service, wait, service,
            max(0.0, self.wait_squares / self.completed - wait ** 2),
            max(0.0, self.service_squares / self.completed - service ** 2),
        )


@dataclass
class Statistics:
    sources: list[SourceStats]
    busy_time: list[float]
    queue_area: float = 0.0
    last_time: float = 0.0
    rejected: int = 0
    completed: int = 0
    start_times: dict[int, float] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)

    def advance_to(self, now: float, devices: list[Device], queued: int):
        elapsed = now - self.last_time
        for device in devices:
            if device.current is not None:
                self.busy_time[device.id - 1] += elapsed
        self.queue_area += queued * elapsed
        self.last_time = now
        self.actions.clear()

    def record_arrival(self, request: Request):
        self.sources[request.source_id - 1].arrived += 1
        self.actions.append(f"И{request.source_id} сгенерировал {request.label}")

    def record_rejection(self, request: Request, now: float):
        source = self.sources[request.source_id - 1]
        source.rejected += 1
        source.rejected_wait_sum += now - request.arrival_at
        self.rejected += 1
        self.actions.append(f"{request.label} вытеснена из буфера (И{request.source_id}, 503 OVERLOAD)")

    def record_enqueue(self, request: Request, result: OfferResult, pointer: int):
        self.actions.append(f"{request.label} записана в ячейку {result.slot}, указатель → {pointer}")

    def record_dequeue(self, request: Request):
        self.actions.append(f"{request.label} выбрана из буфера по FIFO")

    def record_start(self, request: Request, device_id: int, now: float):
        self.start_times[request.id] = now
        self.actions.append(f"{request.label} передана на IMG-{device_id}")

    def record_completion(self, request: Request, device_id: int, now: float):
        start = self.start_times.pop(request.id)
        wait = start - request.arrival_at
        service = now - start
        source = self.sources[request.source_id - 1]
        source.completed += 1
        source.wait_sum += wait
        source.wait_squares += wait * wait
        source.service_sum += service
        source.service_squares += service * service
        self.completed += 1
        self.actions.append(f"IMG-{device_id} завершил {request.label}")


@dataclass
class SelectionDispatcher:
    buffer: RingBuffer
    pool: DevicePool
    calendar: EventCalendar
    stats: Statistics

    def find_free_device(self) -> Device | None:
        return self.pool.first_free()

    def start_on(self, request: Request, device: Device, now: float):
        finish_at = self.pool.start(device, request, now)
        self.stats.record_start(request, device.id, now)
        self.calendar.schedule(EventKind.COMPLETION, finish_at, device.id)

    def dispatch_waiting(self, now: float):
        while self.buffer.count:
            device = self.find_free_device()
            if device is None:
                return
            request = self.buffer.poll_oldest()
            self.stats.record_dequeue(request)
            self.start_on(request, device, now)


@dataclass
class AdmissionDispatcher:
    buffer: RingBuffer
    selection: SelectionDispatcher
    sources: list[Source]
    stats: Statistics

    def accept(self, request: Request, now: float):
        device = self.selection.find_free_device()
        if device is not None:
            self.selection.start_on(request, device, now)
            return
        result = self.buffer.offer(request)
        if result.evicted is not None:
            self.stats.record_rejection(result.evicted, now)
            self.sources[result.evicted.source_id - 1].notify_rejected(result.evicted)
        self.stats.record_enqueue(request, result, self.buffer.pointer)


class Simulation:
    def __init__(self, config: Config, request_limit: int, *, seed: int | None = None):
        if request_limit < 1:
            raise ValueError("Лимит заявок должен быть не меньше 1.")
        rng = Random(config.seed if seed is None else seed)
        self.sources = [
            Source(i, rate, Random(rng.getrandbits(64)))
            for i, rate in enumerate(config.source_rates, 1)
        ]
        self.pool = DevicePool([
            Device(i, config.tmin, config.tmax, Random(rng.getrandbits(64)))
            for i in range(1, config.device_count + 1)
        ])
        self.buffer = RingBuffer(config.buffer_capacity)
        self.calendar = EventCalendar()
        self.stats = Statistics([SourceStats() for _ in self.sources], [0.0] * config.device_count)
        self.selection = SelectionDispatcher(self.buffer, self.pool, self.calendar, self.stats)
        self.admission = AdmissionDispatcher(self.buffer, self.selection, self.sources, self.stats)
        self.now = 0.0
        self.generated = 0
        self.request_limit = request_limit
        for source in self.sources:
            self.calendar.schedule(EventKind.ARRIVAL, source.next_arrival(self.now), source.id)

    @property
    def finished(self) -> bool:
        return not self.calendar.events

    def step(self):
        event = self.calendar.pop_next()
        self.now = event.time
        self.stats.advance_to(self.now, self.pool.devices, self.buffer.count)
        if event.kind == EventKind.ARRIVAL:
            source = self.sources[event.target_id - 1]
            request = source.generate(self.generated, self.now)
            self.stats.record_arrival(request)
            self.generated += 1
            if self.generated == self.request_limit:
                self.calendar.stop_arrivals()
            else:
                self.calendar.schedule(EventKind.ARRIVAL, source.next_arrival(self.now), source.id)
            self.admission.accept(request, self.now)
        else:
            request, result_uri = self.pool.finish(event.target_id)
            self.stats.record_completion(request, event.target_id, self.now)
            self.sources[request.source_id - 1].notify_completed(request, result_uri)
            self.selection.dispatch_waiting(self.now)

        assert self.generated == (self.stats.completed + self.stats.rejected
                                  + self.buffer.count + len(self.stats.start_times))

    def run(self):
        while not self.finished:
            self.step()

    @property
    def refusal_rate(self) -> float:
        return self.stats.rejected / self.generated if self.generated else 0.0


@dataclass(frozen=True)
class ExperimentResult:
    model: Simulation
    converged: bool


class ExperimentRunner:
    initial_requests = 100
    relative_error = 0.1
    confidence_quantile = 1.643

    def run_automatic(self, config: Config, max_requests: int) -> ExperimentResult:
        if max_requests < 2 * self.initial_requests:
            raise ValueError("Верхний предел автоподбора должен быть не меньше 200.")
        rng = Random(config.seed)
        size = self.initial_requests
        previous = None
        while True:
            model = Simulation(config, size, seed=rng.getrandbits(64))
            model.run()
            rate = model.refusal_rate
            if rate > 0:
                required = ceil(self.confidence_quantile ** 2 * (1 - rate)
                                / (rate * self.relative_error ** 2))
                if previous and size >= required and abs(rate - previous) / previous < self.relative_error:
                    return ExperimentResult(model, True)
            else:
                required = size * 2

            if size == max_requests:
                return ExperimentResult(model, False)

            if rate == 0 or (previous and abs(rate - previous) / previous >= self.relative_error):
                required = max(required, size * 2)
            size = min(max_requests, max(size, required))
            previous = rate
