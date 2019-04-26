from marshmallow import Schema, fields
import datetime
import logging

logger = logging.getLogger(__name__)


class TimestampField(fields.Integer):
    default_error_messages = {
        'invalid': 'Not a valid integer.',
        'invalid_date': 'Not a valid DateTime.'
    }

    def _validated(self, value):
        """Format the value or raise a :exc:`ValidationError` if an error occurs."""
        try:
            number = self._format_num(value)
            if number is None:
                return None

            if 946684800 < number < 1893456000:  # 2000 to 2030 allowed
                return number
            else:
                self.fail('invalid_date')
        except (TypeError, ValueError) as err:
            self.fail('invalid')

    def _serialize(self, value, attr, obj):
        if isinstance(value, datetime.datetime):
            return int((value - datetime.datetime(1970, 1, 1)).total_seconds())
        if isinstance(value, datetime.date):
            return int((value - datetime.datetime(1970, 1, 1)).total_seconds())
        return self._validated(value)

    def _deserialize(self, value, attr, data):
        number = self._validated(value)
        dt = datetime.datetime.utcfromtimestamp(number)
        return dt


class BaseSchema(Schema):
    def __deepcopy__(self, memo):
        # XXX: Flask-RESTplus makes unnecessary data copying, while
        # marshmallow.Schema doesn't support deepcopyng.
        return self
    # SKIP_VALUE = None

    def dump(self, obj, **kwargs):
        d = super().dump(obj, **kwargs)
        if len(d.errors) > 0:
            logger.warning("Marshmallow Error:" + str(d))
        return d


class DateTimeNoTimezone(fields.DateTime):
    def _deserialize(self, value, attr, data, **kwargs):
        dt = super()._deserialize(value, attr, data, **kwargs)
        return dt.replace(tzinfo=None)

    def _serialize(self, value, attr, obj, **kwargs):
        if isinstance(value, str):
            return value
        data = super()._serialize(value, attr, obj, **kwargs)
        if data is not None and "+" in data:
            data = data.split("+")[0]
        return data


class CycleEventSchema(BaseSchema):
    _id = fields.Str(description="internal id")
    animal_id = fields.Str(description="animal id")
    event_ts = DateTimeNoTimezone(description="event timestamp")
    create_ts = DateTimeNoTimezone(description="create timestamp")
    event_type = fields.Str(description="event type")
    information = fields.Dict(description="information")
    active = fields.Boolean(description="active")

    # event specific parameter
    cycle_length = fields.Integer(description="length of the current cycle")
    reason = fields.Str(description="length of the current cycle")
    days_to_calving = fields.Integer(description="days from the insemination to the calving")
    expected_calving_date = DateTimeNoTimezone(description="expected calving date")
    insemination_date = DateTimeNoTimezone(description="insemination date")
    pregnant = fields.Boolean(description="if the pregnancy resutl was positive (true) or negative (false)")
    number = fields.Integer(description="the number of calvings")
    late_abort = fields.Boolean(description="when the cow is lactating after an abort")
    icar_key = fields.Str(description="icar key")
    diagnosis_key = fields.Str(description="non icar diagnosis key")
    diagnosis_key_type = fields.Str(description="type of the diagnisis key")


class AnimalStateInfoSchema(BaseSchema):
    state = fields.Str(description="animal cycle state")
    last_insemination_date = DateTimeNoTimezone(
        description="last insemination timestamp")
    last_heat_date = DateTimeNoTimezone(description="last heat timestamp")
    last_calving_date = DateTimeNoTimezone(
        description="last calving date timestamp")
    expected_calving_date = DateTimeNoTimezone(
        description="expected calving date")
    lactation_number = fields.Integer(description="lactation number")
    milk_yield = fields.Integer(description="milk yield")

    cycle_length = fields.Integer(description="cycle length in days")
    insemination_window_start = DateTimeNoTimezone(
        description="start of insenimation window")
    insemination_window_optimal_end = DateTimeNoTimezone(
        description="end of insenimation window")
    insemination_window_end = DateTimeNoTimezone(
        description="end of insenimation window")

    last_calving_detection = DateTimeNoTimezone(
        description="Ts of last detected calving")
    dry_off_date = DateTimeNoTimezone(
        description="Ts of the dry off date")


class AnimalCycleSchema(AnimalStateInfoSchema):
    events = fields.Nested(CycleEventSchema, many=True,
                           description="events of a animal cycle")


class AnimalLifeCycleSchema(AnimalStateInfoSchema):
    cycles = fields.Nested(AnimalCycleSchema, many=True, attribute="cycles_with_padding",
                           description="cycles of a animal life cycle")


class AnimalIdSchema(BaseSchema):
    _id = fields.Str(description="internal id")


class AnimalListContentSchema(BaseSchema):
    name = fields.Str("list name")
    _ids = fields.List(fields.Str(), description="internal animal ids")


class AnimalListNamesSchema(BaseSchema):
    list_names = fields.List(fields.Str(), description="animal list names")


class SimpleAnimalSchema(BaseSchema):
    _id = fields.Str(description="internal id")
    organisation_id = fields.Str(description="organisation id")
    group_id = fields.Str(description="group id")
    name = fields.Str(description="animal name")
    display_name = fields.Str(description="display name")
    metadata = fields.Dict(description="additional information",
                           attribute="animal_metadata")
    tags = fields.List(fields.Str(), description="animal tags")
    mark = fields.Str(description="animal mark")
    official_id = fields.Str(description="official id")
    sensor = fields.Str(description="current sensor id",
                        attribute="current_device_id")
    # sensors = fields.List(fields.Str(), description="animal tags", attribute="device_ids")
    archived = fields.Boolean(description="archiv flag")
    created_at = TimestampField(description="created at")
    lactation_status = fields.Str(description="status of the cow")


class SimpleAnimalSchemaV2(BaseSchema):
    _id = fields.Str(description="internal id")
    name = fields.Str(description="animal name")
    mark = fields.Str(description="animal mark")
    official_id = fields.Str(description="official id")
    display_name = fields.Str(description="display name")

    organisation_id = fields.Str(description="organisation id")
    group_id = fields.Str(
        description="*DEPRECATED* group id, use location in future")
    location = fields.Str(description="Location of the animal")

    sensor = fields.Str(description="*DEPRECATED* current device id",
                        attribute="current_device_id")
    current_device_id = fields.Str(description="current device id",
                                   attribute="current_device_id")

    archived = fields.Boolean(description="archiv flag")
    created_at = DateTimeNoTimezone(description="created at")
    birthday = DateTimeNoTimezone(description="birthday")
    race = fields.Str(description="race")
    mixed_race = fields.List(fields.Dict(), description="mixed_race")
    official_id_rule = fields.Str(description="rule to apply on the official id")

    tags = fields.List(fields.Str(), description="animal tags")

    # sensors = fields.List(fields.Str(), description="animal tags", attribute="device_ids")

    lactation_status = fields.Str(description="*DEPRECATED* status of the cow")


class IntegrationAnimalSchemaV2(SimpleAnimalSchemaV2):
    last_readout = DateTimeNoTimezone(description="last readout timestamp")

    class Meta:
        exclude = ('sensor', )


class AnimalsModulesSchema(BaseSchema):
    animal_id = fields.Str(description="animal id")
    valid_from = DateTimeNoTimezone(description="valid from timestamp")
    active_modules = fields.Dict(description="active modules")


class AnimalSchemaV2(SimpleAnimalSchemaV2):

    lifecycle = fields.Nested(AnimalLifeCycleSchema,
                              description="animal life cycle information", attribute="lifecycle")
    modules = fields.Nested(AnimalsModulesSchema,
                            description="Current module of the animal", attribute="modules")


class AnimalSchemaFirestore(AnimalSchemaV2):
    class Meta:
        exclude = ('heats', 'lactations')


class AnimalStateInfoSchemaInternal(BaseSchema):
    state = fields.Str(description="animal cycle state")
    last_insemination_date = TimestampField(
        description="last insemination timestamp")
    last_heat_date = TimestampField(description="last heat timestamp")
    last_calving_date = TimestampField(
        description="last calving date timestamp")
    expected_calving_date = TimestampField(description="expected calving date")

    cycle_length = fields.Integer(description="cycle length in days")
    insemination_window_start = TimestampField(
        description="start of insenimation window")
    insemination_window_end = TimestampField(
        description="end of insenimation window")

    last_calving_detection = TimestampField(
        description="Ts of last detected calving")


class CycleEventSchemaInternal(BaseSchema):
    _id = fields.Str(description="internal id")
    animal_id = fields.Str(description="animal id")
    event_ts = TimestampField(description="event timestamp")
    create_ts = TimestampField(description="create timestamp")
    event_type = fields.Str(description="event type")
    information = fields.Dict(description="information")
    active = fields.Boolean(description="active")


class AnimalCycleSchemaInternal(AnimalStateInfoSchemaInternal):
    events = fields.Nested(CycleEventSchemaInternal, many=True,
                           description="events of a animal cycle")


class AnimalLifeCycleSchemaInternal(AnimalStateInfoSchemaInternal):
    cycles = fields.Nested(AnimalCycleSchemaInternal, many=True,
                           description="cycles of a animal life cycle")


class AnimalSchemaV2Internal(AnimalSchemaV2):
    created_at = TimestampField(description="created at")
    heats = fields.String(attribute="deprecated")
    lactations = fields.String(attribute="deprecated")
    lifecycle = fields.Nested(AnimalLifeCycleSchemaInternal,
                              description="animal life cycle information", attribute="lifecycle")
    metadata = fields.Dict(description="additional information",
                           attribute="animal_metadata")


class AnimalDevicesSchema(BaseSchema):
    animal_id = fields.Str(description="animal id")
    device_id = fields.Str(description="device id")
    ts = TimestampField(description="ts")
