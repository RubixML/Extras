<?php

namespace Rubix\ML\Tests\Persisters;

use Rubix\ML\Encoding;
use Rubix\ML\Persisters\FlysystemV2;
use Rubix\ML\Persisters\Persister;
use League\Flysystem\Filesystem;
use League\Flysystem\InMemory\InMemoryFilesystemAdapter;
use PHPUnit\Framework\TestCase;

/**
 * @group Persisters
 * @covers \Rubix\ML\Persisters\Flysystem
 */
class FlysystemV2Test extends TestCase
{
    /**
     * The path to the test file.
     *
     * @var string
     */
    protected const PATH = __DIR__ . '/test.model';

    /**
     * @var \Rubix\ML\Persisters\FlysystemV2
     */
    protected $persister;

    /**
     * @before
     */
    protected function setUp() : void
    {
        if (!interface_exists('League\Flysystem\FilesystemOperator')) {
            $this->markTestSkipped();
        }

        $filesystem = new Filesystem(new InMemoryFilesystemAdapter());

        $this->persister = new FlysystemV2(self::PATH, $filesystem, false);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(FlysystemV2::class, $this->persister);
        $this->assertInstanceOf(Persister::class, $this->persister);
    }

    /**
     * @test
     */
    public function saveLoad() : void
    {
        $encoding = new Encoding("Bitch, I'm for real!");

        $this->persister->save($encoding);

        $encoding = $this->persister->load();

        $this->assertInstanceOf(Encoding::class, $encoding);
    }
}
