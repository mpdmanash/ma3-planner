// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: simresult_msg.proto

#include "simresult_msg.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace MA3 {
class SimResultDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<SimResult> _instance;
} _SimResult_default_instance_;
}  // namespace MA3
static void InitDefaultsscc_info_SimResult_simresult_5fmsg_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::MA3::_SimResult_default_instance_;
    new (ptr) ::MA3::SimResult();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_SimResult_simresult_5fmsg_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_SimResult_simresult_5fmsg_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_simresult_5fmsg_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_simresult_5fmsg_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_simresult_5fmsg_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_simresult_5fmsg_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::MA3::SimResult, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::MA3::SimResult, success_),
  PROTOBUF_FIELD_OFFSET(::MA3::SimResult, edge_cost_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::MA3::SimResult)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::MA3::_SimResult_default_instance_),
};

const char descriptor_table_protodef_simresult_5fmsg_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\023simresult_msg.proto\022\003MA3\"/\n\tSimResult\022"
  "\017\n\007success\030\001 \001(\010\022\021\n\tedge_cost\030\002 \001(\001b\006pro"
  "to3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_simresult_5fmsg_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_simresult_5fmsg_2eproto_sccs[1] = {
  &scc_info_SimResult_simresult_5fmsg_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_simresult_5fmsg_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_simresult_5fmsg_2eproto = {
  false, false, descriptor_table_protodef_simresult_5fmsg_2eproto, "simresult_msg.proto", 83,
  &descriptor_table_simresult_5fmsg_2eproto_once, descriptor_table_simresult_5fmsg_2eproto_sccs, descriptor_table_simresult_5fmsg_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_simresult_5fmsg_2eproto::offsets,
  file_level_metadata_simresult_5fmsg_2eproto, 1, file_level_enum_descriptors_simresult_5fmsg_2eproto, file_level_service_descriptors_simresult_5fmsg_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_simresult_5fmsg_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_simresult_5fmsg_2eproto)), true);
namespace MA3 {

// ===================================================================

class SimResult::_Internal {
 public:
};

SimResult::SimResult(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:MA3.SimResult)
}
SimResult::SimResult(const SimResult& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&edge_cost_, &from.edge_cost_,
    static_cast<size_t>(reinterpret_cast<char*>(&success_) -
    reinterpret_cast<char*>(&edge_cost_)) + sizeof(success_));
  // @@protoc_insertion_point(copy_constructor:MA3.SimResult)
}

void SimResult::SharedCtor() {
  ::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
      reinterpret_cast<char*>(&edge_cost_) - reinterpret_cast<char*>(this)),
      0, static_cast<size_t>(reinterpret_cast<char*>(&success_) -
      reinterpret_cast<char*>(&edge_cost_)) + sizeof(success_));
}

SimResult::~SimResult() {
  // @@protoc_insertion_point(destructor:MA3.SimResult)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void SimResult::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void SimResult::ArenaDtor(void* object) {
  SimResult* _this = reinterpret_cast< SimResult* >(object);
  (void)_this;
}
void SimResult::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SimResult::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const SimResult& SimResult::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_SimResult_simresult_5fmsg_2eproto.base);
  return *internal_default_instance();
}


void SimResult::Clear() {
// @@protoc_insertion_point(message_clear_start:MA3.SimResult)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ::memset(&edge_cost_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&success_) -
      reinterpret_cast<char*>(&edge_cost_)) + sizeof(success_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SimResult::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // bool success = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          success_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // double edge_cost = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 17)) {
          edge_cost_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* SimResult::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:MA3.SimResult)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // bool success = 1;
  if (this->success() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(1, this->_internal_success(), target);
  }

  // double edge_cost = 2;
  if (!(this->edge_cost() <= 0 && this->edge_cost() >= 0)) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(2, this->_internal_edge_cost(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:MA3.SimResult)
  return target;
}

size_t SimResult::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:MA3.SimResult)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // double edge_cost = 2;
  if (!(this->edge_cost() <= 0 && this->edge_cost() >= 0)) {
    total_size += 1 + 8;
  }

  // bool success = 1;
  if (this->success() != 0) {
    total_size += 1 + 1;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void SimResult::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:MA3.SimResult)
  GOOGLE_DCHECK_NE(&from, this);
  const SimResult* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<SimResult>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:MA3.SimResult)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:MA3.SimResult)
    MergeFrom(*source);
  }
}

void SimResult::MergeFrom(const SimResult& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:MA3.SimResult)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (!(from.edge_cost() <= 0 && from.edge_cost() >= 0)) {
    _internal_set_edge_cost(from._internal_edge_cost());
  }
  if (from.success() != 0) {
    _internal_set_success(from._internal_success());
  }
}

void SimResult::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:MA3.SimResult)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SimResult::CopyFrom(const SimResult& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:MA3.SimResult)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SimResult::IsInitialized() const {
  return true;
}

void SimResult::InternalSwap(SimResult* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(SimResult, success_)
      + sizeof(SimResult::success_)
      - PROTOBUF_FIELD_OFFSET(SimResult, edge_cost_)>(
          reinterpret_cast<char*>(&edge_cost_),
          reinterpret_cast<char*>(&other->edge_cost_));
}

::PROTOBUF_NAMESPACE_ID::Metadata SimResult::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace MA3
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::MA3::SimResult* Arena::CreateMaybeMessage< ::MA3::SimResult >(Arena* arena) {
  return Arena::CreateMessageInternal< ::MA3::SimResult >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>