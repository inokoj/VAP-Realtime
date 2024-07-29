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

    const wavesurferLeft = initializeWaveSurfer('#waveform_left', 'rgb(255, 165, 0)', 'rgba(255, 165, 0, 0.5)', '/audio/left');
    const wavesurferRight = initializeWaveSurfer('#waveform_right', 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)', '/audio/right');

    const synchronizePlayPause = (play) => {
        if (play) {
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
        switch (event.code) {
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

    fetch('/data')
        .then(response => response.json())
        .then(data => {
            const time = data.map(item => parseFloat(item.time_sec));

            const ctxNow = document.getElementById('probabilityChartNow').getContext('2d');
            const ctxFuture = document.getElementById('probabilityChartFuture').getContext('2d');

            const prepareProbData = (probData) => {
                let above = [], below = [];
                probData.forEach((value, index) => {
                    if (value > 0.5) {
                        above.push({ x: time[index], y: value });
                        below.push({ x: time[index], y: 0.5 });
                    } else {
                        above.push({ x: time[index], y: 0.5 });
                        below.push({ x: time[index], y: value });
                    }
                });
                return { above, below };
            };

            const createProbChart = (ctx, probData, label) => {
                const { above, below } = prepareProbData(probData);
                return new Chart(ctx, {
                    type: 'line',
                    data: {
                        datasets: [
                            {
                                label: `${label} (above 0.5)`,
                                data: above,
                                borderColor: 'rgba(255, 165, 0, 1)',
                                backgroundColor: 'rgba(255, 165, 0, 0.5)',
                                pointRadius: 0,
                                fill: { target: { value: 0.5 }, above: 'rgba(255, 165, 0, 0.5)', below: 'rgba(255, 165, 0, 0.5)' },
                                borderWidth: 1,
                                segment: {
                                    borderColor: ctx => ctx.p0.parsed.y > 0.5 ? 'rgba(255, 165, 0, 1)' : 'rgba(0, 0, 255, 1)',
                                    backgroundColor: ctx => ctx.p0.parsed.y > 0.5 ? 'rgba(255, 165, 0, 0.5)' : 'rgba(0,0,255,0.5)'
                                }
                            },
                            {
                                label: `${label} (below 0.5)`,
                                data: below,
                                borderColor: 'rgba(0, 0, 255, 1)',
                                backgroundColor: 'rgba(0, 0, 255, 0.5)',
                                pointRadius: 0,
                                fill: { target: { value: 0.5 }, above: 'rgba(0, 0, 255, 0.5)', below: 'rgba(0, 0, 255, 0.5)' },
                                borderWidth: 1,
                                segment: {
                                    borderColor: ctx => ctx.p0.parsed.y > 0.5 ? 'rgba(255, 165, 0, 1)' : 'rgba(0, 0, 255, 1)',
                                    backgroundColor: ctx => ctx.p0.parsed.y > 0.5 ? 'rgba(255, 165, 0, 0.5)' : 'rgba(0, 0, 255, 0.5)'
                                }
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
                                },
                                min: 0,
                                max: 1
                            }
                        },
                        plugins: {
                            legend: {
                                display: false,
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
                        }
                    }
                });
            };

            const probNow = data.map(item => parseFloat(item['p_now(0=left)']));
            const probFuture = data.map(item => parseFloat(item['p_future(0=left)']));

            const chartNow = createProbChart(ctxNow, probNow, 'Probability Now');
            const chartFuture = createProbChart(ctxFuture, probFuture, 'Probability Future');

            const updateChartMax = (chart, wavesurfer) => {
                const duration = wavesurfer.getDuration();
                chart.options.scales.x.max = duration;
                chart.update();
            };

            const updateAnnotation = (chart, time) => {
                if (chart.options.plugins.annotation.annotations.line1) {
                    chart.options.plugins.annotation.annotations.line1.value = time;
                    chart.update('none');
                }
            };

            const synchronizeSeek = (wavesurferFrom, wavesurferTo) => {
                const currentTime = wavesurferFrom.getCurrentTime();
                const duration = wavesurferTo.getDuration();
                const isPlaying = wavesurferFrom.isPlaying();
         
                wavesurferTo.pause();
                wavesurferTo.seekTo(currentTime / duration);
                updateAnnotation(chartNow, currentTime);
                updateAnnotation(chartFuture, currentTime);
                if (isPlaying) {
                    setTimeout(() => wavesurferTo.play(), 0);
                }
            };

            wavesurferLeft.on('ready', () => updateChartMax(chartNow, wavesurferLeft));
            wavesurferRight.on('ready', () => updateChartMax(chartFuture, wavesurferRight));

            wavesurferLeft.on('interaction', () => synchronizeSeek(wavesurferLeft, wavesurferRight));
            wavesurferRight.on('interaction', () => synchronizeSeek(wavesurferRight, wavesurferLeft));

            wavesurferLeft.on('audioprocess', () => {
                const currentTime = wavesurferLeft.getCurrentTime();
                updateAnnotation(chartNow, currentTime);
                updateAnnotation(chartFuture, currentTime);
            });
        });
});