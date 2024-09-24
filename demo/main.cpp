#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama2.h"
#include <chrono>
#include <cuda_runtime.h> // 用于获取显存占用

int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::InputPos);

  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    if (next == model.get_eos()) {
      break;
    }
    std::string word;
    if (is_prompt) {
      next = tokens.at(pos + 1);
      word = model.decode(next);
    } else {
      word = model.decode(next);
    }

    if (need_output) {
      printf("%s ", word.c_str());
      fflush(stdout);
    }
    pos += 1;
  }
  return std::min(pos, total_steps);
}

size_t memory_usage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem - free_mem;
    //   printf("GPU Memory Used: %d MB / %d MB\n", 
    //          (total_mem - free_mem) >> 20, 
    //          total_mem >> 20);
}

int main(int argc, char* argv[]) {

    size_t occ_mem = memory_usage();

    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path ";
        return -1;
    }
    const char* checkpoint_path = argv[1];  // e.g. out/model.bin
    const char* tokenizer_path = argv[2];
    
    // 初始化模型
    model::LLama2Model model(tokenizer_path, checkpoint_path, true);
    auto init_status = model.init(base::DeviceType::DeviceCUDA);
    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
    }

    size_t new_occ_mem = memory_usage();
    printf("GPU Memory Used: %ld MB \n", (new_occ_mem - occ_mem) >> 20);

    const std::string& sentence = "hello I am an AnHui University student, and my name is Dawiix.";

    // warmup
    int steps = generate(model, sentence, 256, false);
    steps = generate(model, sentence, 256, false);
    
    // 推理速度测量
    auto start = std::chrono::steady_clock::now();
    steps = generate(model, sentence, 128, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\ntokens/s:%lf\n", static_cast<double>(steps) / duration);
    
    // 测试显存占用
    memory_usage();

    return 0;
}
