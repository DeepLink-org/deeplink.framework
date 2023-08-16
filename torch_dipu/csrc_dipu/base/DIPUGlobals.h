#pragma once
namespace dipu {

void initResource();

void releaseAllResources();

bool is_in_bad_fork();
void poison_fork();

};