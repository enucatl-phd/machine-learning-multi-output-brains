require "rake"

desc "fisher_0"
file "data/fisher_0.npz" => ["fisher_score_saved.py", "data/median_0_0.npz", "data/median_1_0.npz", "data/sd_0_0.npz", "data/sd_1_0.npz"] do |f|
  sh "python #{f.source} --median0 #{f.prerequisites[1]} --median1 #{f.prerequisites[2]} --sd0 #{f.prerequisites[3]} --sd1 #{f.prerequisites[4]} #{f.name}"
end
