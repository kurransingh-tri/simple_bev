./avctl run src/backend/tools:download_data -- -- --data_source=av \
--mission-id=7616883548368623693_15377212525588467702 \
--min-time-ns=1748615303070826000 \
--max-time-ns=1748615427000099000 \
--output_dir_name="/tmp/lyftbags" \
--topics=/avs/mission,/avs/config,/calibration,\
/arene/localization/ensd_localization,/arene/perception/perception_online_local_map,\
/arene/perception/perception_tracks,/arene/perception/prediction_obstacles,\
/arene/perception/traffic_light_output,/avs/reflection,\
/avs/screenshot,/avs/screenshot_meta,/avs/slog_events,\
/bosch_radar_0/points,/bosch_radar_0/targets,\
/bosch_radar_1/points,/bosch_radar_1/targets,\
/bosch_radar_4/points,/bosch_radar_4/targets,\
/bosch_radar_7/points,/bosch_radar_7/targets,\
/bosch_radar_8/points,/bosch_radar_8/targets,\
/cam0/image_compressed,/cam0/image_compressed_meta,\
/cam1/image_compressed,/cam1/image_compressed_meta,\
/cam2/image_compressed,/cam2/image_compressed_meta,\
/cam3/image_compressed,/cam3/image_compressed_meta,\
/cam4/image_compressed,/cam4/image_compressed_meta,\
/cam5/image_compressed,/cam5/image_compressed_meta,\
/cam6/image_compressed,/cam6/image_compressed_meta,\
/cam7/image_compressed,/cam7/image_compressed_meta,\
/cam8/image_compressed,/cam8/image_compressed_meta,\
/compressed_semantic_map,\
/compressed_semantic_map_ensd,\
/control/controller_command_viz,\
/control/driver_controls,\
/control/envelope_controller_command,\
/control/envelope_controller_data,\
/driving/perception/perception_tracks,\
/driving/perception/traffic_light_output,\
/local_projection,\
/localization/motion_state_local,\
/localization/transform_local_ecef,\
/map_fusion/map_fusion_debug_message,\
/map_fusion/map_fusion_debug_message_viz,\
/map/map_id,\
/map/map_id_ensd,\
/map/route_name,\
/novatel/gnsspos,\
/novatel/inspva,\
/novatel/rawimu,\
/perception/perception_obstacles,\
/perception/spindata,\
/perception/spindata_nova,\
/planning,\
/planning/lane_change_status_output,\
/planning/local_scene_debug_adas_ecu,\
/planning/planner_context,\
/planning/trajectory_candidates,\
/planning/trajectory_planning_problem,\
/planning/turn_indicator_output,\
/scope_manager/feature_scope,\
/scope_manager/obstacle_risk,\
/ublox/navsol,\
/ublox/navsvinfo,\
/vehicle_report/autonomy_event,\
/vehicle_report/autonomy_state,\
/vehicle_report/gear,\
/vehicle_report/is_stationary,\
/vehicle_report/speed,\
/vehicle_report/steering,\
/vehicle_report/turn_indicator_stalk_report,\
/vehicle_report/vehicle_state_summary,\
/vehicle_report/wheel_speed \
