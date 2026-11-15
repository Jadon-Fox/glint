# Project 7: typed, independently addressable refinement controls.
# This is a narrow protocol/scaffold for #117, not a predictive scheduler.
# Explicitly preserve the distinction between transfer shape, timing,
# selection, application and buffer lifetime.
struct RefineConfig(Copyable, Movable):
    var transfer_granularity: Int32
    var prefetch_distance_layers: Int32
    var selection_rule: Int32
    var application_rule: Int32
    var staging_slots: Int32

    # transfer: 0 whole-layer, 1 tile, 2 K-major channel group.
    # selection: 0 all, 1 current activation, 2 random, 3 static hot,
    #            4 previous token, 5 previous layer.
    # application: 0 full-table direct GEMV, 1 parent+delta accumulate.
    # A config is not proof an option has been implemented.
    def __init__(
        out self,
        transfer_granularity: Int32,
        prefetch_distance_layers: Int32,
        selection_rule: Int32,
        application_rule: Int32,
        staging_slots: Int32,
    ):
        self.transfer_granularity = transfer_granularity
        self.prefetch_distance_layers = prefetch_distance_layers
        self.selection_rule = selection_rule
        self.application_rule = application_rule
        self.staging_slots = staging_slots

    def validate(self) raises:
        if self.transfer_granularity < 0 or self.transfer_granularity > 2:
            raise Error("unsupported transfer granularity")
        if self.prefetch_distance_layers < 0:
            raise Error("negative prefetch distance")
        if self.selection_rule < 0 or self.selection_rule > 5:
            raise Error("unsupported selection rule")
        if self.application_rule < 0 or self.application_rule > 1:
            raise Error("unsupported application rule")
        if self.staging_slots < 2 or self.staging_slots > 16:
            raise Error("staging slots must be 2..16")

    def slot_for_layer(self, layer: Int) -> Int:
        return layer % Int(self.staging_slots)

    def require_implemented_whole_layer_full(self) raises:
        self.validate()
        if self.transfer_granularity != 0 or self.selection_rule != 0 or self.application_rule != 0:
            raise Error("requested policy not implemented by whole-layer GLINT reference")
        if self.prefetch_distance_layers != 1:
            raise Error("reference scheduler supports only one-layer lookahead")
        if self.staging_slots != 2:
            raise Error("reference scheduler supports only two slots")

# An operation-level request must preserve its own deadline, not infer it
# from the stream granularity or the refinement precision.
struct RefineRequest(Copyable, Movable):
    var layer: Int32
    var linear: Int32
    var first_channel: Int32
    var channel_count: Int32
    var precision_bits: Int32
    var transfer_bytes: Int
    var deadline_op: Int32

    def __init__(
        out self,
        layer: Int32,
        linear: Int32,
        first_channel: Int32,
        channel_count: Int32,
        precision_bits: Int32,
        transfer_bytes: Int,
        deadline_op: Int32,
    ):
        self.layer = layer
        self.linear = linear
        self.first_channel = first_channel
        self.channel_count = channel_count
        self.precision_bits = precision_bits
        self.transfer_bytes = transfer_bytes
        self.deadline_op = deadline_op

    def validate(self) raises:
        if self.layer < 0 or self.linear < 0 or self.linear > 3:
            raise Error("invalid consumer")
        if self.first_channel < 0 or self.channel_count < 0:
            raise Error("invalid channel interval")
        if self.precision_bits < 1 or self.precision_bits > 16:
            raise Error("invalid precision")
        if self.transfer_bytes < 0:
            raise Error("negative bytes")

def main() raises:
    var cfg = RefineConfig(0, 1, 0, 0, 2)
    cfg.require_implemented_whole_layer_full()
    if cfg.slot_for_layer(3) != 1:
        raise Error("slot allocation regression")
    var req = RefineRequest(1, 2, 0, 64, 4, 32, 3)
    req.validate()
    print("P7_REFINE_CONFIG_CONTRACT_PASS")
