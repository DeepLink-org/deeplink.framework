#pragma once
namespace dipu {

void initResource();

void releaseAllResources();

// TODO(caikun): move to helpfunc.h
bool is_in_bad_fork();
void poison_fork();

};