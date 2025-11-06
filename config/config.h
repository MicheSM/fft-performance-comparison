// __asm__ rbit in reverse_bits
// irrilevante per le versioni vettorizzate, che usano l'intrinsic svrbit
// irrilevante comunque per armclang++ che riconosce il bit hack
// __asm__ non piace al vettorizzatore, ma non riesce lo stesso a vettorizzare quel ciclo
#define USE_RBIT_ASM

// non stampare il risultato del fft
#define DONTPRINT

// stampa i tempi di esecuzione
#define ENABLE_TIMERS

// stampa solo i numeri di microsecondi, più comodo da parsare
#define COMPACT_OUTPUT
