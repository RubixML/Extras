<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\BooleanArray;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Generator;

use function serialize;
use function unpack;
use function hash;
use function exp;

/**
 * Deduplicator
 *
 * Removes duplicate records from a dataset while the records are in flight. Deduplicator uses a memory-efficient
 * Bloom filter to probabilistically identify records that have already been seen before.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Deduplicator implements Extractor
{
    /**
     * The base hashing function.
     *
     * @var string
     */
    protected const HASH_FUNCTION = 'crc32b';

    /**
     * The base iterator.
     *
     * @var iterable<array>
     */
    protected iterable $iterator;

    /**
     * The size of the bloom filter.
     *
     * @var int
     */
    protected int $size;

    /**
     * The number of hash functions.
     *
     * @var int
     */
    protected int $numHashes;

    /**
     * A fixed size boolean array.
     *
     * @var \Rubix\ML\BooleanArray
     */
    protected \Rubix\ML\BooleanArray $bitmap;

    /**
     * The number of records that have been processed so far.
     */
    protected int $n = 0;

    /**
     * @param iterable<mixed[]> $iterator
     * @param int $size
     * @param int $numHashes
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(iterable $iterator, int $size, int $numHashes = 3)
    {
        if ($size < 1) {
            throw new InvalidArgumentException('Size must be'
                . " greater than 1, $size given.");
        }

        if ($numHashes < 1) {
            throw new InvalidArgumentException('Number of hashes'
                . " must be greater than 1, $numHashes given.");
        }

        $this->iterator = $iterator;
        $this->size = $size;
        $this->numHashes = $numHashes;
        $this->bitmap = new BooleanArray($size);
    }

    /**
     * Return the probability of mistakenly dropping a unique record.
     *
     * @return float
     */
    public function falsePositiveRate() : float
    {
        return (1.0 - exp(-$this->numHashes * $this->n / $this->size)) ** $this->numHashes;
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Generator
    {
        foreach ($this->iterator as $record) {
            $token = serialize($record);

            if (!$this->existsOrInsert($token)) {
                yield $record;
            }

            ++$this->n;
        }
    }

    /**
     * Does a value exist in the Bloom filter? If so, return true or insert and return false.
     *
     * @param string $value
     * @return bool
     */
    protected function existsOrInsert(string $value) : bool
    {
        $hashes = $this->hashes($value);

        $exists = true;

        foreach ($hashes as $hash) {
            if (!$this->bitmap[$hash]) {
                $this->bitmap[$hash] = true;

                if ($exists) {
                    $exists = false;
                }
            }
        }

        return $exists;
    }

    /**
     * Return an array of hashes from a given string.
     *
     * @param string $value
     * @return list<int>
     */
    protected function hashes(string $value) : array
    {
        $hashes = [];

        for ($i = 0; $i < $this->numHashes; ++$i) {
            $digest = hash(self::HASH_FUNCTION, "$i:$value");

            /** @var int[] $bytes */
            $bytes = unpack('S*', $digest);

            $hash = 0;

            if (PHP_INT_SIZE === 8) {
                $hash |= $bytes[1] << 48;
                $hash |= $bytes[2] << 32;
                $hash |= $bytes[3] << 16;
                $hash |= $bytes[4];
            } else {
                $hash |= $bytes[1] << 16;
                $hash |= $bytes[2];
            }

            $hash %= $this->size;

            $hashes[] = $hash;
        }

        return $hashes;
    }
}
