// Copyright (c) 2023, DeepLink.
#include "DIPUGeneratorImpl.h"

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <c10/util/logging_is_not_google_glog.h>

#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

namespace dipu {

// TODO(global) - Stop using non const global variables.

// Ensures call initDIPUGenerator once
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::once_flag dipu_init_flag;

// Total number of dipu in the system.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static int64_t num_dipu;

// Ensures default_gens_dipu is initialized once.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::deque<std::once_flag> dipu_gens_init_flag;

// Default, global dipu generators, one per dipu.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::vector<at::Generator> default_gens_dipu;

/*
 * Populates the global variables related to DIPU generators
 * Warning: this function must only be called once!
 */
static void initDIPUGenerator() {
  num_dipu = devproxy::getDeviceCount();
  dipu_gens_init_flag.resize(num_dipu);
  default_gens_dipu.resize(num_dipu);
}

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 */
at::Generator& getDefaultDIPUGenerator(at::DeviceIndex device_index) {
  std::call_once(dipu_init_flag, initDIPUGenerator);

  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = devproxy::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_dipu);
  }
  std::call_once(dipu_gens_init_flag[idx], [&] {
    default_gens_dipu[idx] = vendorMakeGenerator(idx);
    default_gens_dipu[idx].seed();
  });
  return default_gens_dipu[idx];
}

/**
 * Utility to create a DIPUGeneratorImpl. Returns a shared_ptr
 */
at::Generator createDIPUGenerator(at::DeviceIndex device_index) {
  std::call_once(dipu_init_flag, initDIPUGenerator);
  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = devproxy::current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_dipu, "The device_index is invalid.");
  auto generator = vendorMakeGenerator(idx);
  auto gen_impl = at::check_generator<DIPUGeneratorImpl>(generator);
  gen_impl->set_current_seed(c10::default_rng_seed_val);
  return generator;
}

/**
 * DIPUGeneratorImpl class implementation
 */
DIPUGeneratorImpl::DIPUGeneratorImpl(at::DeviceIndex device_index)
    : c10::GeneratorImpl{at::Device(dipu::DIPU_DEVICE_TYPE, device_index),
                         at::DispatchKeySet(dipu::DIPU_DISPATCH_KEY)},
      offset_(0),
      state_need_reset_(true) {}

/**
 * Sets the seed to be used by MTGP
 *
 * See Note [Acquire lock when using random generators]
 */
void DIPUGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  state_need_reset_ = true;
}

/**
 * Gets the current seed of DIPUGeneratorImpl.
 */
uint64_t DIPUGeneratorImpl::current_seed() const { return seed_; }

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 */
uint64_t DIPUGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/*
 * Gets the DeviceType of DIPUGeneratorImpl.
 * Used for type checking during run time.
 */
at::DeviceType DIPUGeneratorImpl::device_type() {
  return dipu::DIPU_DEVICE_TYPE;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<DIPUGeneratorImpl> DIPUGeneratorImpl::clone() const {
  return std::shared_ptr<DIPUGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
DIPUGeneratorImpl* DIPUGeneratorImpl::clone_impl() const {
  auto gen = dynamic_cast<DIPUGeneratorImpl*>(
      createDIPUGenerator(this->device().index()).unsafeReleaseGeneratorImpl());
  TORCH_CHECK(gen != nullptr);
  gen->set_current_seed(this->seed_);
  auto state = this->state_;
  const auto& state_clone = state.clone();
  gen->set_state(*state_clone.getIntrusivePtr());
  gen->set_state_flag(this->state_need_reset_);
  return gen;
}

/**
 * get state
 *
 * See Note [Acquire lock when using random generators]
 */
c10::intrusive_ptr<c10::TensorImpl> DIPUGeneratorImpl::get_state() const {
  if (state_need_reset_) {
    update_state();
  }
  auto state_clone = state_.clone();
  return state_clone.getIntrusivePtr();
}

/**
 * set state flag
 * See Note [Acquire lock when using random generators]
 */
void DIPUGeneratorImpl::set_state_flag(bool flag) { state_need_reset_ = flag; }

/**
 * get rng state
 *
 **/
at::Tensor get_rng_state(at::DeviceIndex idx) {
  auto gen = getDefaultDIPUGenerator(idx);
  auto gen_impl = at::get_generator_or_default<DIPUGeneratorImpl>(
      gen, getDefaultDIPUGenerator());
  std::lock_guard<std::mutex> lock(gen_impl->mutex_);
  auto state_ptr = gen_impl->get_state();
  auto state = at::Tensor(std::move(state_ptr));
  return state;
}

/**
 * set rng state
 *
 **/
void set_rng_state(at::DeviceIndex idx, at::Tensor const& state) {
  auto pointer = state.getIntrusivePtr();
  TORCH_CHECK_NOTNULL(pointer);

  auto gen = getDefaultDIPUGenerator(idx);
  auto gen_impl = at::get_generator_or_default<DIPUGeneratorImpl>(
      gen, getDefaultDIPUGenerator());
  std::lock_guard<std::mutex> lock(gen_impl->mutex_);
  gen_impl->set_state(*pointer);
}

/**
 * set manual seed
 *
 **/
void manual_seed(at::DeviceIndex idx, uint64_t seed) {
  auto gen = getDefaultDIPUGenerator(idx);
  auto gen_impl = at::get_generator_or_default<DIPUGeneratorImpl>(
      gen, getDefaultDIPUGenerator());
  std::lock_guard<std::mutex> lock(gen_impl->mutex_);
  gen_impl->set_current_seed(seed);
}

/**
 * seed
 *
 **/
void seed(at::DeviceIndex idx) {
  auto gen = getDefaultDIPUGenerator(idx);
  auto gen_impl = at::get_generator_or_default<DIPUGeneratorImpl>(
      gen, getDefaultDIPUGenerator());
  std::lock_guard<std::mutex> lock(gen_impl->mutex_);
  gen_impl->seed();
}

/**
 * initial seed
 *
 **/
uint64_t initial_seed(at::DeviceIndex idx) {
  auto defaultGenerator = getDefaultDIPUGenerator(idx);
  auto seed = defaultGenerator.current_seed();
  return seed;
}

void releaseAllGenerator() { default_gens_dipu.clear(); }

}  // namespace dipu
