create table games(
    id int auto_increment not null primary key,
    name varchar(100) not null
);
create table cars(
	id int auto_increment not null primary key,
    name varchar(100) not null,
    game_id int,
    foreign key (game_id) references games(id) on delete set null
);
create table tracks(
	id int auto_increment not null primary key,
    name varchar(100),
    game_id int,
    foreign key (game_id) references games(id) on delete set null
);
create table stages(
	id int auto_increment not null primary key,
    name varchar(100),
    track_id int,
    foreign key (track_id) references tracks(id) on delete cascade
);
create table car_setups(
	id int auto_increment not null primary key,
    name varchar(100),
    setup_data JSON not null,
    track_id int,
    stage_id int,
    game_id int,
    foreign key (game_id) references games(id) on delete set null,
    foreign key (track_id) references tracks(id) on delete set null,
    foreign key (stage_id) references stages(id) on delete set null
);