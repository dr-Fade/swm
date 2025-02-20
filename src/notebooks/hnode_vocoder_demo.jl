### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d4f88d0c-df2a-4ae9-8d1c-72da02ecfff6
begin
	using Pkg; Pkg.activate("../../")
	using Lux, BSON, PlutoUI, Random, CairoMakie
	# include("../../src/hnode_vocoder/hnode_vocoder_training.jl")
end;

# ╔═╡ 03787f2a-f9f6-4b47-9c13-81dbf47efae5
using OrdinaryDiffEq

# ╔═╡ 95ca5fe1-850b-4a6a-a5a4-779834654099
md"# Дослідження проблеми синтезу мови за допомогою нейромережевих моделей"

# ╔═╡ eb6dce90-7062-4e51-a636-1caab809cbea
html"""
<p style="text-align:right;">
	<b>Здобувач:</b> Кошель Євген Васильович
</p>
<p style="text-align:right;">
	<b>Науковий керівник:</b> проф., д-р фіз.-мат. наук Білозьоров Василь Євгенович
</p>
"""

# ╔═╡ 9b503408-bd79-4630-aff2-93c0f297e4c8
md"""
# Огляд предметної області та попередніх досліджень
"""

# ╔═╡ 7ea066b2-2233-48c5-969c-4dd3d3a4b8bc
md"""
# Динамічні системи
"""

# ╔═╡ e498f97a-ea15-4971-ba1f-1f993c2815ff
md"""
*Система* -- це сутність, яка сприймається як єдиний об'єкт, що складається із взаємопов'язаних елементів.

Система, стан якої змінюється з часом називається *динамічною*. В іншому випадку система *статична*.

Прикладом динамічної системи є модель математичного маятника:
"""

# ╔═╡ b0b029c4-1fec-43e9-adc5-afd351440ca6
md"l = $(@bind l PlutoUI.NumberField(1:10))"

# ╔═╡ 2da15045-0886-4453-878d-6b8fb83b7ac8
md"g = $(@bind g PlutoUI.NumberField(0.1f0:0.01f0:10f0, default = 9.7f0))"

# ╔═╡ a988b67d-f6de-415a-bfea-71a2b8fb963b
begin
	θmin = -10
	θmax = 10
	md"θ₀ = $(@bind θ₀ NumberField(θmin:0.1f0:θmax; default=rand(Float32)))"
end

# ╔═╡ f4189f3d-eab6-4c65-aad2-0c6054561ffe
begin
	vmin = -10
	vmax = 10
	md"v₀ = $(@bind v₀ NumberField(vmin:0.1f0:vmax; default=rand(Float32)))"
end

# ╔═╡ f9f5e286-f684-4590-84e8-9854ca5bffe3
# ╠═╡ disabled = true
#=╠═╡
begin
	function pendulum(du, u, p, t)
	    du[1] = u[2]
	    du[2] = -g * sin(u[1]) / l
	end
	u0 = [θ₀; v₀]
	tspan = (0f0, 10f0)
	saveat = tspan[1]:0.001f0:tspan[2]
	prob = ODEProblem(pendulum, u0, tspan, saveat = saveat)
	sol = solve(prob, Tsit5())
	(
		scatter([θ₀], [v₀], label="(θ₀, v₀)", legend=false);
		plot!(
			sol[1,:],
			sol[2,:],
			label = "trajectory",
			xlabel = "θ₀",
			ylabel = "v₀",
			xlim=(θmin,θmax),
			ylim=(vmin,vmax),
			line_z = 1:size(sol[:])[1],
			linewidth = 2,
			c = :inferno,
			size = (1200,800)
		)
	)
end
  ╠═╡ =#

# ╔═╡ 523579a9-2df2-4957-8fe3-3ed2e958d468
md"""
# Вилучення визначних характеристик зі звуку
"""

# ╔═╡ b36489aa-cd14-4f06-8263-62308fcc5c62
md"Директорія: $(@bind sound_dir Select(get_dirs_with_sound(\"../samples/\")))"

# ╔═╡ fc6d76fd-cf81-4194-8023-839666cdafda
md"Файл: $(@bind sound_filename Select(get_all_sound_files(sound_dir)))"

# ╔═╡ 88ce0070-9bb8-4699-b03a-4e991d5585a0
md"Додатковий шум: $(@bind noise PlutoUI.Slider(0f0:0.001f0:0.1f0; show_value = true))"

# ╔═╡ 4379f8de-557b-4aaf-881c-751363be75f1
md"""
### Завантажений сигнал
"""

# ╔═╡ e037b430-0aa6-4aa6-8d62-c0c64fdf70b0
md"### Визначні характеристики"

# ╔═╡ b871bd84-ade0-454d-967c-08b2cec7fb59
md"""
Нормалізувати $(@bind normalize_features PlutoUI.CheckBox())
"""

# ╔═╡ 88258951-1495-42d5-9ffd-076a73577ead
md"""
# Моделювання сегмента сигналу
"""

# ╔═╡ 5b100402-83a0-48f6-9da0-24542aa1a788
md"Тривалість сегмента: $(@bind segment_length_s PlutoUI.NumberField(0.07f0:0.01f0:1f0)) seconds."

# ╔═╡ 4c51224f-e241-4bcb-b874-9e4bceaad0dc
md"""
### Генерування сигналу з визначних характеристик
"""

# ╔═╡ 722d7edf-0bf1-40a2-be5d-a8bb86049ea8
md"""
## Imports and includes
"""

# ╔═╡ b87d0038-0e68-4431-bd39-1f6beb973dea
model = begin
	model_filename = "hnode_vocoder_params.bson"
	model_state_filename = "hnode_vocoder_state.bson"

	n = Integer(FEATURE_EXTRACTION_SAMPLE_RATE ÷ F0_CEIL) ÷ 2
	sample_rate = FEATURE_EXTRACTION_SAMPLE_RATE
	hNODEVocoder(sample_rate; n=n);
end

# ╔═╡ e35c86a8-5450-4e71-9b36-6c7aeb16569f
begin
	original_sound = get_sound(
		sound_filename;
		target_sample_rate = model.sample_rate,
	)
	test_data = get_training_data(model, original_sound; β=noise)
	input_sound = vcat(eachcol(test_data.input[1:model.n,:])...)
	sound_duration_s = length(input_sound) / FEATURE_EXTRACTION_SAMPLE_RATE |> Float32
	ylim=(min(input_sound...), max(input_sound...))
	md"""
	* Тривалість сигналу: $(round(length(input_sound) / model.sample_rate; digits=2)) секунд.
	* Кількість сегментів: $(size(test_data.features.loudness)[2]).
	* Відстань між сегментами: $(model.n) семплів.
	"""
end

# ╔═╡ cb38c405-cb53-47ed-89d8-c01ac2d7fc7a
let
	fig = Figure(; size = (1200,200))
	lines!(
		Axis(fig[1,1]),
		(1:length(input_sound)) ./ FEATURE_EXTRACTION_SAMPLE_RATE,
		input_sound
	)
	fig
end

# ╔═╡ 8c6ea712-812f-4d05-84bf-1ec22bed78aa
md"Початок сегмента: $(@bind segment_start_s PlutoUI.NumberField(0.1f0:Float32(model.n / FEATURE_EXTRACTION_SAMPLE_RATE):sound_duration_s)) second."

# ╔═╡ 6a29b2d4-cb2e-45ea-9bca-08acea07c973
begin
	start_sample = max(segment_start_s * FEATURE_EXTRACTION_SAMPLE_RATE |> floor |> Int, 1)
	encoder_samples_n = segment_length_s * FEATURE_EXTRACTION_SAMPLE_RATE |> floor |> Int
	samples_n = encoder_samples_n - model.feature_scanner.cell.input_n + 2
	end_sample = start_sample + samples_n

	start_frame = max(start_sample ÷ model.n, 1)
	frames_n = samples_n ÷ model.n
	end_frame = start_frame + frames_n - 1

	segment_frames = test_data.input[:,start_frame:end_frame]
	segment = reduce(vcat, segment_frames[1:model.n,:])
	autoencoder_segment = reduce(vcat, test_data.input[1:model.n,start_frame:(end_frame+Int(3*model.sample_rate÷F0_FLOOR)÷model.n+1)])
	
	segment_features = (
		test_data.features.f0s[:, start_frame:end_frame],
		test_data.features.loudness[:, start_frame:end_frame],
		test_data.features.mfccs[:, start_frame:end_frame]
	)

	let
		fig = Figure(; size = (1200,800))
		lines!(
			(ax = Axis(fig[1,1], title = "Сегмент"); ylims!(minimum(input_sound), maximum(input_sound)); ax),
			segment
		)
		lines!(
			(ax = Axis(fig[2,1], title = "Оцінки фундаментальної частоти"); ylims!(0,F0_CEIL); ax),
			segment_features[1][1,:]
		)
		lines!(
			(ax = Axis(fig[3,1], title = "Впевненість оцінки"); ylims!(0,1); ax),
			segment_features[1][2,:]
		)
		lines!(
			Axis(fig[4,1], title = "Гучність"),
			segment_features[2][:]
		)
		heatmap!(
			Axis(fig[5,1], title = "MFCC"),
			segment_features[3]',
			colormap = :blues
		)
		fig
	end
end

# ╔═╡ 99ba00c8-8b0a-44f9-9c82-b8127ff2fdad
if isfile(model_filename)
    @info "loading params from file"
    BSON.@load model_filename ps
	BSON.@load model_state_filename st
	f0s_batch_norm_st, loudness_batch_norm_st, mfccs_batch_norm_st = (l.layer_1 for l in st.control.layer_1)
else
    @info "creating new params"
	rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
	st = Lux.testmode(st)
end;

# ╔═╡ ed7c5761-6d64-4b3f-b7dc-ddc07cb5bf93
let
	CairoMakie.activate!(type = "png")
	title_sound = get_sound("../samples/LibriSpeech/demo/103-1241-0019.flac")[5000:15000]
	title_training_data = get_training_data(model, title_sound)
	title_latent_xs = time_series_to_latent(model.encoder, ps.encoder, st.encoder, title_sound, model.stream_filter.cell.n)
	title_latent_n = size(title_latent_xs)[1]
	fig = Figure(; size = (1200, 700))

	lines!(
		Axis(fig[1, 1]),
		(1:title_latent_n) ./ FEATURE_EXTRACTION_SAMPLE_RATE,
		title_sound[1:title_latent_n],
		colormap = :rainbow
	)
	lines!(
		Axis3(
			fig[2, 1],
			aspect=(4,1,1),
			xlabel = "t",
			ylabel = "",
			zlabel = "",
			perspectiveness = 1,
			xspinesvisible = false,
			zspinesvisible = false,
			yspinesvisible = false,
			azimuth = -pi/2,
			elevation = pi / 10,
		),
		(1:length(title_latent_xs[:,1])) ./ FEATURE_EXTRACTION_SAMPLE_RATE,
		title_latent_xs[:,1],
		title_latent_xs[:,2],
		color = (1:length(title_latent_xs[:,1])),
		colormap = :prism
	)
	rowsize!(fig.layout, 1, Fixed(200))
	fig
	# plot(
	# 	# plot(title_sound),
	# 	plot3d(
	# 		1:length(title_latent_xs[:,1]),
	# 		title_latent_xs[:,1],
	# 		title_latent_xs[:,2],
	# 		linewidth = 1,
	# 		line_z = 1:size(title_latent_xs)[1],
	# 		c = :rainbow,
	# 		autosize=false,
	# 		field_of_view = 90,
	# 		size=(1200,400),
	# 		legend = false,
	# 	)
	# )
end

# ╔═╡ 53c99c91-2b36-45c8-9087-2899c49c98a2
let
	fig = Figure(; size = (1200,600))
	lines!(
		Axis(fig[1,1], title = "Оцінки фундаментальної частоти"),
		test_data.features.f0s[1,:] ./ if normalize_features F0_CEIL else 1 end
	)
	lines!(
		Axis(fig[2,1], title = "Впевненість оцінки"),
		test_data.features.f0s[2,:]
	)
	lines!(
		Axis(fig[1,2], title = "Гучність"),
		if normalize_features
			(test_data.features.loudness[:] .- loudness_batch_norm_st.running_mean) ./ loudness_batch_norm_st.running_var
		else
			test_data.features.loudness[:]
		end
	)
	heatmap!(
		Axis(fig[2,2], title = "MFCC"),
		(if normalize_features
			(test_data.features.mfccs .- mfccs_batch_norm_st.running_mean) ./ mfccs_batch_norm_st.running_var
		else
			test_data.features.mfccs
		end)',
		colormap = :blues
	)
	fig
end
# 	plot(test_data.features.f0s[2,:], title = "Впевненість оцінки"),
# 	size=(800,400),
# 	layout = @layout [a;b]
# )

# ╔═╡ 07f89fca-65f5-41a3-be3d-7978f8f47f26
begin
	latent_xs = time_series_to_latent(model.encoder, ps.encoder, st.encoder, autoencoder_segment, model.stream_filter.cell.n)
	decoded = latent_to_time_series(model.decoder, ps.decoder, st.decoder, latent_xs)
	model_output, _ = get_connected_trajectory(model, ps, st, (segment_frames, segment_features))
	let
		fig = Figure(; size = (1200,900))
		let
			ax = Axis(fig[1,1:2], title = "Часова область")
			ylims!(minimum(input_sound), maximum(input_sound))
			lines!(ax, segment, color=:blue, label = "Цільовий сигнал")
			lines!(ax, decoded[1:length(segment)], color=:red, label = "Вивід автокодувальника")
			lines!(ax, model_output, color=:green, label = "Результат інтегрування моделі")
			axislegend(ax)
		end
		lim = 2
		let
			x, y, z = latent_xs[:,1], latent_xs[:,2], latent_xs[:,3]
			lims = (
				min(-lim, minimum(x), minimum(y), minimum(z)),
				max(lim, maximum(x), maximum(y), maximum(z))
			)
			lines!(
				(
					ax = Axis3(fig[2,1], perspectiveness = 0.1, title = "Проєкція перших трьох латентних змінних");
					xlims!(lims...); ylims!(lims...); zlims!(lims...);
					ax
				),
				x, y, z,
				color = 1:size(latent_xs)[1],
				colormap = :jet1
			)
		end
		let
			x, y, z = latent_xs[:,4], latent_xs[:,5], latent_xs[:,6]
			lims = (
				min(-lim, minimum(x), minimum(y), minimum(z)),
				max(lim, maximum(x), maximum(y), maximum(z))
			)
			lines!(
				(
					ax = Axis3(fig[2,2], perspectiveness = 0.1, title = "Проєкція четвертої, п'ятої, та шостої латентних змінних");
					xlims!(lims...); ylims!(lims...); zlims!(lims...);
					ax
				),
				x, y, z,
				color = 1:size(latent_xs)[1],
				colormap = :jet1
			)
		end
		fig
	end

	# plot(
	# 	(
	# 		plot(segment, label = "target", linewidth = 2, ylim=ylim);
	# 		plot!(decoded, label = "decoded", title="Autoencoder's output", linewidth = 2)
	# 	),
	# 	plot(
	# 		[
	# 			plot(
	# 				latent_xs[:,i],
	# 				latent_xs[:,i+1],
	# 				latent_xs[:,i+3],
	# 				linewidth = 2,
	# 				line_z = 1:size(latent_xs)[1],
	# 				c = :jet1,
	# 				legend = false,
	# 			)
	# 			for i ∈ 1:2:LATENT_DIMS-3]...,
	# 	),
	# 	size=(1200,600)
	# )
end

# ╔═╡ 3b69e450-1a17-4639-bcd7-40fcd195b3c7
plot(
	plot(st.control.layer_1.mfccs.layer_1.running_var, title="MFCC running_var"),
	plot(st.control.layer_1.mfccs.layer_1.running_mean, title="MFCC running mean"),
	size=(1200,400)
)

# ╔═╡ 57887708-7bad-4d83-b6fa-d9f44b131fd3
plot(
	heatmap(ps.encoder.layer_2.weight),
	heatmap(ps.encoder.layer_3.weight),
	size=(1200,1200),
	layout = @layout[a;b]
)

# ╔═╡ 5b8681ad-1752-40de-9dd1-8e6d28e90551
HTML("""
<!-- the wrapper span -->
<div>
	<button id="myrestart" href="#">Restart</button>
	
	<script>
		const div = currentScript.parentElement
		const button = div.querySelector("button#myrestart")
		const cell= div.closest('pluto-cell')
		console.log(button);
		button.onclick = function() { restart_nb() };
		function restart_nb() {
			console.log("Restarting Notebook");
		        cell._internal_pluto_actions.send(                    
		            "restart_process",
                            {},
                            {
                                notebook_id: editor_state.notebook.notebook_id,
                            }
                        )
		};
	</script>
</div>
""")

# ╔═╡ 5f582b39-226c-44ee-976a-5480f2fe6c3a
html"""<style>
main {
    max-width: 1000px;
}
"""

# ╔═╡ Cell order:
# ╟─95ca5fe1-850b-4a6a-a5a4-779834654099
# ╟─ed7c5761-6d64-4b3f-b7dc-ddc07cb5bf93
# ╟─eb6dce90-7062-4e51-a636-1caab809cbea
# ╟─9b503408-bd79-4630-aff2-93c0f297e4c8
# ╟─7ea066b2-2233-48c5-969c-4dd3d3a4b8bc
# ╟─e498f97a-ea15-4971-ba1f-1f993c2815ff
# ╟─b0b029c4-1fec-43e9-adc5-afd351440ca6
# ╟─2da15045-0886-4453-878d-6b8fb83b7ac8
# ╟─a988b67d-f6de-415a-bfea-71a2b8fb963b
# ╟─f4189f3d-eab6-4c65-aad2-0c6054561ffe
# ╟─03787f2a-f9f6-4b47-9c13-81dbf47efae5
# ╠═f9f5e286-f684-4590-84e8-9854ca5bffe3
# ╟─523579a9-2df2-4957-8fe3-3ed2e958d468
# ╟─b36489aa-cd14-4f06-8263-62308fcc5c62
# ╟─fc6d76fd-cf81-4194-8023-839666cdafda
# ╟─88ce0070-9bb8-4699-b03a-4e991d5585a0
# ╟─e35c86a8-5450-4e71-9b36-6c7aeb16569f
# ╟─4379f8de-557b-4aaf-881c-751363be75f1
# ╟─cb38c405-cb53-47ed-89d8-c01ac2d7fc7a
# ╟─e037b430-0aa6-4aa6-8d62-c0c64fdf70b0
# ╟─b871bd84-ade0-454d-967c-08b2cec7fb59
# ╟─53c99c91-2b36-45c8-9087-2899c49c98a2
# ╟─88258951-1495-42d5-9ffd-076a73577ead
# ╟─8c6ea712-812f-4d05-84bf-1ec22bed78aa
# ╟─5b100402-83a0-48f6-9da0-24542aa1a788
# ╟─6a29b2d4-cb2e-45ea-9bca-08acea07c973
# ╟─4c51224f-e241-4bcb-b874-9e4bceaad0dc
# ╠═07f89fca-65f5-41a3-be3d-7978f8f47f26
# ╟─722d7edf-0bf1-40a2-be5d-a8bb86049ea8
# ╠═d4f88d0c-df2a-4ae9-8d1c-72da02ecfff6
# ╟─b87d0038-0e68-4431-bd39-1f6beb973dea
# ╠═99ba00c8-8b0a-44f9-9c82-b8127ff2fdad
# ╟─3b69e450-1a17-4639-bcd7-40fcd195b3c7
# ╟─57887708-7bad-4d83-b6fa-d9f44b131fd3
# ╟─5b8681ad-1752-40de-9dd1-8e6d28e90551
# ╟─5f582b39-226c-44ee-976a-5480f2fe6c3a
