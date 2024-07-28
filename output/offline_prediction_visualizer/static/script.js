document.addEventListener('DOMContentLoaded', function () {
    const generateCacheBuster = () => `?_=${new Date().getTime()}`;

    const initializeWaveSurfer = (container, waveColor, progressColor, url) => {
        const wavesurfer = WaveSurfer.create({
            container,
            waveColor,
            progressColor
        });
        wavesurfer.load(`${url}${generateCacheBuster()}`);
        return wavesurfer;
    };

    const wavesurferLeft = initializeWaveSurfer('#waveform_left', 'violet', 'purple', '/audio/left');
    const wavesurferRight = initializeWaveSurfer('#waveform_right', 'lightblue', 'blue', '/audio/right');

    const synchronizePlayPause = (play) => {
        if (play) {
            const currentTime = wavesurferLeft.getCurrentTime();
            const duration = wavesurferRight.getDuration();
            wavesurferRight.seekTo(currentTime / duration);
            wavesurferLeft.play();
            wavesurferRight.play();
        } else {
            wavesurferLeft.pause();
            wavesurferRight.pause();
        }
    };

    const playPauseButton = document.getElementById('playPause');
    const togglePlayPause = () => {
        const isPlaying = wavesurferLeft.isPlaying();
        synchronizePlayPause(!isPlaying);
        playPauseButton.innerHTML = isPlaying ? '<i class="fas fa-play"></i>Play' : '<i class="fas fa-pause"></i>Pause';
    }

    playPauseButton.addEventListener('click', togglePlayPause);

    const setPlaybackRate = (speed) => {
        wavesurferLeft.setPlaybackRate(speed);
        wavesurferRight.setPlaybackRate(speed);
    };

    document.querySelectorAll('[id^=speed-]').forEach(button => {
        button.addEventListener('click', () => {
            const speed = parseFloat(button.id.split('-')[1]);
            setPlaybackRate(speed);
        });
    });

    document.addEventListener('keydown', (event) => {
        switch(event.code) {
            case 'Space':
                event.preventDefault();
                togglePlayPause();
                break;
            case 'Digit1':
                setPlaybackRate(1);
                break;
            case 'Digit2':
                setPlaybackRate(2);
                break;
            default:
                break;
        }
    });

    const synchronizeSeek = (wavesurferFrom, wavesurferTo) => {
        const currentTime = wavesurferFrom.getCurrentTime();
        const duration = wavesurferTo.getDuration();
        const isPlaying = wavesurferFrom.isPlaying();

        wavesurferTo.pause();
        wavesurferTo.seekTo(currentTime / duration);
        if (isPlaying) {
            setTimeout(() => wavesurferTo.play(), 0); // Small timeout to ensure the seek is processed first
        }
    };

    fetch('/data')
        .then(response => response.json())
        .then(data => {
            const time = data.map(item => item.time_sec);
            const createProbChart = (ctx, p_now, p_future, colors) => new Chart(ctx, {
                type: 'line',
                data: {
                    labels: time,
                    datasets: [
                        {
                            label: 'Probability Now',
                            data: p_now,
                            borderColor: colors.now,
                            fill: false,
                            radius: 1
                        },
                        {
                            label: 'Probability Future',
                            data: p_future,
                            borderColor: colors.future,
                            fill: false,
                            radius: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            title: {
                                display: true,
                                text: 'Time (s)'
                            },
                            min: 0,
                            max: 0
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Probability'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'bottom',
                            align: 'end'
                        },
                        annotation: {
                            annotations: {
                                line1: {
                                    type: 'line',
                                    borderColor: 'purple',
                                    borderWidth: 2,
                                    mode: 'vertical',
                                    scaleID: 'x',
                                    value: 0,
                                    label: {
                                        enabled: true,
                                        content: 'Current Time',
                                        position: 'top'
                                    }
                                }
                            }
                        }
                    },
                    onClick: (event, elements) => {
                        if (elements.length > 0) {
                            const element = elements[0];
                            const time = chart.data.labels[element.index];
                            const duration = wavesurfer.getDuration();
                            wavesurfer.seekTo(time / duration);
                        }
                    }
                }
            });

            const ctxLeft = document.getElementById('probabilityChartLeft').getContext('2d');
            const ctxRight = document.getElementById('probabilityChartRight').getContext('2d');

            const chartLeft = createProbChart(ctxLeft, data.map(item => item['p_now(0=left)']), data.map(item => item['p_future(0=left)']), { now: 'red', future: 'blue' });
            const chartRight = createProbChart(ctxRight, data.map(item => item['p_now(1=right)']), data.map(item => item['p_future(1=right)']), { now: 'green', future: 'orange' });

            const updateChartMax = (chart, wavesurfer) => {
                const duration = wavesurfer.getDuration();
                chart.options.scales.x.max = duration;
                chart.update();
            };

            wavesurferLeft.on('ready', () => updateChartMax(chartLeft, wavesurferLeft));
            wavesurferRight.on('ready', () => updateChartMax(chartRight, wavesurferRight));

            wavesurferLeft.on('interaction', () => synchronizeSeek(wavesurferLeft, wavesurferRight));
            wavesurferRight.on('interaction', () => synchronizeSeek(wavesurferRight, wavesurferLeft));
        });
});