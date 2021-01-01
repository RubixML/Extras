<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function is_string;

/**
 * Bzip2
 *
 * A compression format based on the Burrows–Wheeler algorithm. Bzip2 is slightly smaller than
 * Gzip but is slower and requires more memory.
 *
 * References:
 * [1] J. Tsai. (2006). Bzip2: Format Specification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Bzip2 implements Serializer
{
    /**
     * The size of each block between 1 and 9 where 9 gives the best compression.
     *
     * @var int
     */
    protected $blockSize;

    /**
     * Controls how the compression phase behaves when the input is highly repetitive.
     *
     * @var int
     */
    protected $workFactor;

    /**
     * The base serializer.
     *
     * @var \Rubix\ML\Persisters\Serializers\Serializer
     */
    protected $serializer;

    /**
     * @param int $blockSize
     * @param int $workFactor
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $blockSize = 4, int $workFactor = 0, ?Serializer $serializer = null)
    {
        if (!extension_loaded('bz2')) {
            throw new RuntimeException('Bzip2 extension is not'
                . ' loaded, check PHP configuration.');
        }

        if ($blockSize < 1 or $blockSize > 9) {
            throw new InvalidArgumentException('Block size must'
                . " be between 0 and 9, $blockSize given.");
        }

        if ($serializer instanceof self) {
            throw new InvalidArgumentException('Base serializer'
                . ' must not be an instance of itself.');
        }

        $this->blockSize = $blockSize;
        $this->workFactor = $workFactor;
        $this->serializer = $serializer ?? new Native();
    }

    /**
     * Serialize a persistable object and return the data.
     *
     * @param \Rubix\ML\Persistable $persistable
     * @return \Rubix\ML\Encoding
     */
    public function serialize(Persistable $persistable) : Encoding
    {
        $encoding = $this->serializer->serialize($persistable);

        $data = bzcompress((string) $encoding, $this->blockSize, $this->workFactor);

        if (!is_string($data)) {
            throw new RuntimeException('Failed to compress data.');
        }

        return new Encoding($data);
    }

    /**
     * Unserialize a persistable object and return it.
     *
     * @param \Rubix\ML\Encoding $encoding
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(Encoding $encoding) : Persistable
    {
        $data = bzdecompress((string) $encoding);

        if (!is_string($data)) {
            throw new RuntimeException('Failed to decompress data.');
        }

        return $this->serializer->unserialize(new Encoding($data));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Bzip2 (block size: {$this->blockSize}, work factor: {$this->workFactor},"
            . " serializer: {$this->serializer})";
    }
}
