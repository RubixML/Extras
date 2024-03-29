<?php

namespace Rubix\ML\Benchmarks\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ISRLU;

/**
 * @Groups({"ActivationFunctions"})
 * @BeforeMethods({"setUp"})
 */
class ISRLUBench
{
    /**
     * @var \Tensor\Matrix
     */
    protected $z;

    /**
     * @var \Tensor\Matrix
     */
    protected $computed;

    /**
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ISRLU
     */
    protected $activationFn;

    public function setUp() : void
    {
        $this->z = Matrix::uniform(500, 500);

        $this->computed = Matrix::uniform(500, 500);

        $this->activationFn = new ISRLU();
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function compute() : void
    {
        $this->activationFn->activate($this->z);
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function differentiate() : void
    {
        $this->activationFn->differentiate($this->z, $this->computed);
    }
}
