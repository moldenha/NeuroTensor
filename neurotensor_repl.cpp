#include <cling/Interpreter/Interpreter.h>
#include <cling/Interpreter/Value.h>
#include <iostream>

int main(int argc, char** argv) {
    cling::Interpreter interpreter(argc, argv, nullptr, true);

    interpreter.process("#include <iostream>");
    interpreter.process("#include \"nt/nt.h\"");
    interpreter.process("using namespace nt;");

    std::cout << "Welcome to the NeuroTensor REPL! Type .q to quit.\n";

    std::string line;
    while (std::cout << "nt>> " && std::getline(std::cin, line)) {
        if (line == ".q") break;
        cling::Value result;
        interpreter.process(line, &result);
    }

    return 0;
}

