```bash
.
├─── Makefile                       # compile all source
│                    
├─── README.md                      # project description
│
├─── build/                         # compiled binaries
│   ├─── cooley-tuckey/
│   ├─── element-wise-product/
│   └─── stockham/
│
├─── config/                        # macros
│
├─── data/                           
│   ├─── inputs/                    # arrays of complex vectors
│   ├─── reference-outputs/         # precalculated solutions for correctness testing
│   └─── results/                   
│       ├─── errors/                # algorithmic error
│       │   ├─── cooley-tuckey/
│       │   └─── stockham/
│       ├─── plots/                 
│       ├─── raw-outputs/           # test outputs 
│       │   ├─── cooley-tuckey/
│       │   └─── stockham/
│       ├─── statistics/            # analitical insights
│       │   ├─── cooley-tuckey/
│       │   └─── stockham/
│       └─── times/                 # execution times
│           ├─── cooley-tuckey/
│           └─── stockham/
│
├─── scripts/                        
│   ├─── bash/                      # running tests
│   ├─── matlab/                    # reference output generation
│   └─── python/                    # data analysis and plot generation
│      
└─── src/
    ├─── algorithms/                # algoritms under test
    │   ├─── cooley-tuckey/
    │   ├─── element-wise-product/
    │   └─── stockham/
    └─── utils/                     # timer, input generation, error calculation