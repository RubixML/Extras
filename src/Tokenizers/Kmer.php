<?php

namespace Rubix\ML\Tokenizers;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * K-mer
 *
 * K-mers are substrings of sequences such as DNA containing the bases A, T, C, and G with a length of *k*.
 * They are often used in bioinformatics to represent features of a DNA sequence.
 *
 * !!! note
 *     K-mers that contain invalid bases will not be generated.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Kmer implements Tokenizer
{
    /**
     * The length of tokenized sequences.
     *
     * @var int
     */
    protected int $k;

    /**
     * The number of k-mers that were dropped due to invalid bases.
     *
     * @var int
     */
    protected int $dropped = 0;

    /**
     * @param int $k
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $k = 4)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be'
                . " greater than 1, $k given.");
        }

        $this->k = $k;
    }

    /**
     * Return the number of k-mers that were dropped due to invalid bases.
     *
     * @return int
     */
    public function dropped() : int
    {
        return $this->dropped;
    }

    /**
     * Tokenize a blob of text.
     *
     * @internal
     *
     * @param string $text
     * @return list<string>
     */
    public function tokenize(string $text) : array
    {
        $p = strlen($text) - $this->k;

        $tokens = [];

        for ($i = 0; $i <= $p; ++$i) {
            $token = substr($text, $i, $this->k);

            if (preg_match('/[^ACTG]/', $token, $matches, PREG_OFFSET_CAPTURE)) {
                $skip = 1 + (int) $matches[0][1];

                $i += $skip;

                $this->dropped += $skip;

                continue;
            }

            $tokens[] = $token;
        }

        return $tokens;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "K-mer (k: {$this->k})";
    }
}
